use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fs;
use std::io::{self};
use std::sync::LazyLock;
use std::{env, process};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

#[derive(Debug, Deserialize)]
struct Response {
    choices: Vec<Choice>,
}
#[derive(Debug, Deserialize)]
struct Choice {
    index: usize,
    message: Message,
    finish_reason: Option<String>,
}
#[derive(Debug, Deserialize, Serialize, Clone)]
struct Message {
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
    tool_call_id: Option<String>,
}
#[derive(Debug, Deserialize, Serialize, Clone)]
struct ToolCall {
    id: String,
    r#type: String,
    function: FunctionCall,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct FunctionCall {
    name: String,
    arguments: String,
}

static tools: LazyLock<Vec<Value>> = LazyLock::new(|| {
    vec![json!({
      "type": "function",
      "function": {
        "name": "Read",
        "description": "Read and return the contents of a file",
        "parameters": {
          "type": "object",
          "properties": {
            "file_path": {
              "type": "string",
              "description": "The path to the file to read"
            }
          },
          "required": ["file_path"]
        }
      }
    })]
});

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    let api_key = env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| {
        eprintln!("OPENROUTER_API_KEY is not set");
        process::exit(1);
    });

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key);

    let client = Client::with_config(config);
    let mut messages: Vec<Message> = vec![Message {
        role: "user".to_owned(),
        content: Some(args.prompt),
        tool_calls: None,
        tool_call_id: None,
    }];
    loop {
        #[allow(unused_variables)]
        let response = call_llm(&client, &messages, "anthropic/claude-haiku-4.5").await?;
        let response: Response = serde_json::from_value(response).unwrap();
        let message = &response.choices[0].message;
        messages.push(message.clone());
        match &message.tool_calls {
            None => {
                if let Some(content) = &message.content {
                    println!("{content}");
                    break;
                }
            }
            Some(tool) => {
                let tool_res = execute_tool(&tool[0])?;
                messages.push(Message {
                    role: String::from("tool"),
                    content: Some(tool_res),
                    tool_call_id: Some((&tool[0]).id.clone()),
                    tool_calls: None,
                });
            }
        }
    }

    Ok(())
}

async fn call_llm(
    client: &Client<OpenAIConfig>,
    messages: &Vec<Message>,
    model: &str,
) -> anyhow::Result<Value> {
    let response: Value = client
        .chat()
        .create_byot(json!({
            "messages": messages,
            "model": model,
            "tools":*tools
        }))
        .await?;
    Ok(response)
}

fn execute_tool(tool: &ToolCall) -> io::Result<String> {
    match tool.function.name.as_str() {
        "Read" => read_tool(tool.function.arguments.as_str()),
        _ => Ok("Not found".to_string()),
    }
}

fn read_tool(args: &str) -> io::Result<String> {
    let map: HashMap<String, String> = serde_json::from_str(args).unwrap_or_default();
    let file_path = map
        .get("file_path")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "file path is required"))?;
    let content = fs::read_to_string(file_path)?;
    Ok(content)
}

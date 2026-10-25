use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;
use std::{env, process};

#[derive(Deserialize)]
struct SkillMetadata {
    name: String,
    description: String,
    context: Option<String>,
}

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

static TOOLS: LazyLock<Vec<Value>> = LazyLock::new(|| {
    vec![
        json!({
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
        }),
        json!({
          "type": "function",
          "function": {
            "name": "Write",
            "description": "Write content to a file",
            "parameters": {
              "type": "object",
              "required": ["file_path", "content"],
              "properties": {
                "file_path": {
                  "type": "string",
                  "description": "The path of the file to write to"
                },
                "content": {
                  "type": "string",
                  "description": "The content to write to the file"
                }
              }
            }
          }
        }),
        json!({
          "type": "function",
          "function": {
            "name": "Bash",
            "description": "Execute a shell command",
            "parameters": {
              "type": "object",
              "required": ["command"],
              "properties": {
                "command": {
                  "type": "string",
                  "description": "The command to execute"
                }
              }
            }
          }
        }),
        json!({
          "type": "function",
          "function": {
            "name": "Skill",
            "description": "Load a skill's instructions into the conversation",
            "parameters": {
              "type": "object",
              "required": ["name"],
              "properties": {
                "name": { "type": "string", "description": "The name of the skill to use" },
                "args": { "type": "string", "description": "Optional arguments for the skill" }
              }
            }
          }
        }),
    ]
});
static BASE_URL: LazyLock<String> = LazyLock::new(|| {
    env::var("OPENROUTER_BASE_URL").unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string())
});

static API_KEY: LazyLock<String> = LazyLock::new(|| {
    env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| {
        eprintln!("OPENROUTER_API_KEY is not set");
        process::exit(1);
    })
});
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut messages: Vec<Message> = Vec::new();
    if let Ok(skills) = load_all_skills() {
        messages.push(Message {
            role: String::from("system"),
            content: Some(format!(
                "You have access to the following skills:\n\n-{} \n

                If a skill matches the user's request, call the Skill tool with its name
                and follow the instructions it returns.",
                skills
            )),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    if let Ok(user_content) = evaluate_user_content(args.prompt) {
        messages.push(Message {
            role: "user".to_owned(),
            content: Some(user_content),
            tool_calls: None,
            tool_call_id: None,
        });
    }
    if let Ok(content) = run_agent(&mut messages).await {
        println!("{content}");
    }

    Ok(())
}

async fn run_agent(messages: &mut Vec<Message>) -> Result<String, Box<dyn std::error::Error>> {
    let config = OpenAIConfig::new()
        .with_api_base(BASE_URL.as_str())
        .with_api_key(API_KEY.as_str());
    let client = Client::with_config(config);
    loop {
        #[allow(unused_variables)]
        let response = call_llm(&client, &messages, "anthropic/claude-haiku-4.5").await?;
        let response: Response = serde_json::from_value(response).unwrap();
        let message = &response.choices[0].message;
        messages.push(message.clone());
        match &message.tool_calls {
            None => {
                if let Some(content) = &message.content {
                    return Ok(content.to_owned());
                }
            }
            Some(tool_calls) => {
                for tool in tool_calls {
                    let tool_res = execute_tool(&tool).await?;
                    messages.push(Message {
                        role: String::from("tool"),
                        content: Some(tool_res),
                        tool_call_id: Some((&tool).id.clone()),
                        tool_calls: None,
                    });
                }
            }
        }
    }
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
            "tools":*TOOLS
        }))
        .await?;
    Ok(response)
}

fn load_all_skills() -> Result<String, Box<dyn std::error::Error>> {
    let mut directories = Vec::new();
    let mut result = Vec::new();
    for entry in fs::read_dir(".claude/skills/")? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            directories.push(entry.file_name().to_string_lossy().into_owned());
            let file = PathBuf::from(entry.path()).join("SKILL.md");
            let file_content = read_file(&file).unwrap();
            let rest = file_content
                .strip_prefix("---\n")
                .ok_or("Missing opening front-matter delimiter")?;

            let (front_matter, content) = rest
                .split_once("\n---")
                .ok_or("Missing closing front-matter delimiter")?;

            let metadata: SkillMetadata = serde_yaml::from_str(front_matter)?;
            result.push(format!("{}: {}", metadata.name, metadata.description));
        }
    }
    return Ok(result.join("\n -"));
}

fn evaluate_user_content(prompt: String) -> io::Result<String> {
    if prompt.starts_with("/") {
        let mut skill_args_iter = prompt.split_whitespace();
        let mut skills: Vec<&str> = Vec::new();
        let mut skill_args = Vec::new();
        for arg in skill_args_iter.by_ref() {
            if !arg.starts_with("/") {
                skill_args.push(arg);
                break;
            }
            let arg = &arg[1..];
            let path = PathBuf::from(".claude/skills").join(arg).join("SKILL.md");
            if !path.exists() {
                skill_args.push(arg);
                break;
            }
            skills.push(arg);
        }
        skill_args.extend(skill_args_iter);
        let mut final_content = String::new();
        for skill_name in skills.iter() {
            let file = PathBuf::from(".claude/skills")
                .join(skill_name)
                .join("SKILL.md");
            let input = read_file(&file).unwrap();
            let (_, content) = input.rsplit_once("---").unwrap();
            let mut content = content.to_string();
            if !skill_args.is_empty() {
                for (index, value) in skill_args.iter().enumerate() {
                    content = content.replace(&format!("$ARGUMENTS[{index}]"), value);
                    content = content.replace(&format!("${index}"), value);
                }
                content = content.replace("$ARGUMENTS", &skill_args.join(" "));
            }
            final_content.push_str(
                format!(
                    "Skill: {skill_name} (located at .claude/skills/{skill_name})\n\
                     Paths in the instructions below are relative to .claude/skills/{skill_name}.\n\
                     To execute a bundled script, prepend .claude/skills/{skill_name} to its relative path.\n
                    \n\n
                        Its SKILL.md instructions are included below
                     "
                )
                .as_str(),
            );

            final_content.push_str(&content.trim());
            final_content.push_str("\n");
        }
        return Ok(final_content);
    } else {
        return Ok(prompt);
    }
}

fn read_file(file_path: &PathBuf) -> io::Result<String> {
    let content = fs::read_to_string(file_path)?;
    Ok(content)
}

fn read_tool(args: &str) -> io::Result<String> {
    let map: HashMap<String, String> = serde_json::from_str(args).unwrap_or_default();
    let file_path = map
        .get("file_path")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "file path is required"))?;
    let file = PathBuf::from(file_path);
    read_file(&file)
}

async fn execute_tool(tool: &ToolCall) -> io::Result<String> {
    match tool.function.name.as_str() {
        "Read" => read_tool(&tool.function.arguments),
        "Write" => write_tool(&tool.function.arguments),
        "Bash" => execute_bash(&tool.function.arguments),
        "Skill" => skill_tool(&tool.function.arguments).await,
        name => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unknown tool: {name}"),
        )),
    }
}

async fn skill_tool(args: &str) -> io::Result<String> {
    let map: HashMap<String, String> = serde_json::from_str(args)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
    let skill_name = map
        .get("name")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Skill name is required"))?;

    let args = map.get("args");
    let path = PathBuf::from(".claude/skills")
        .join(skill_name)
        .join("SKILL.md");
    let file_content = read_file(&path)?;
    let rest = file_content.strip_prefix("---\n").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Missing opening front-matter delimiter",
        )
    })?;

    let (front_matter, content) = rest.split_once("\n---").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Missing closing front-matter delimiter",
        )
    })?;

    let metadata: SkillMetadata = serde_yaml::from_str(front_matter).unwrap();
    match metadata.context.as_deref() {
        None => return Ok(file_content),
        Some("fork") => {
            let mut messages: Vec<Message> = vec![Message {
                role: String::from("user"),
                content: Some(file_content.to_owned()),
                tool_call_id: None,
                tool_calls: None,
            }];
            if let Ok(res) = Box::pin(run_agent(&mut messages)).await {
                dbg!(&res);
                return Ok(res);
            }
        }
        Some(_) => return Ok(file_content),
    }
    Ok(String::new())
}

fn write_tool(args: &str) -> io::Result<String> {
    let map: HashMap<String, String> = serde_json::from_str(args)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;

    let file_path = map
        .get("file_path")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "file path is required"))?;
    let content = map
        .get("content")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "content is required"))?;

    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(file_path)?;

    let mut writer = BufWriter::new(file);
    writer.write_all(content.as_bytes())?;
    writer.flush()?;

    Ok("Done".to_string())
}

fn execute_bash(args: &str) -> io::Result<String> {
    let map: HashMap<String, String> = serde_json::from_str(args)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;

    let command = map
        .get("command")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "file path is required"))?;

    let mut iter = command.split_whitespace();
    let cmd_name = iter
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "command is required"))?;
    let cmd_args: Vec<&str> = iter.collect();

    let output = Command::new(cmd_name).args(cmd_args).output()?;

    let bytes = if output.status.success() {
        output.stdout
    } else {
        output.stderr
    };

    String::from_utf8(bytes).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
}

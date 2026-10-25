#!/usr/bin/env bash
cd "$(dirname "$0")/.." && sha256sum data.txt | cut -c1-8
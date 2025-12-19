#!/usr/bin/env python3
"""Lightweight terminal chat client for Ollama."""

import argparse
import json
import os
import sys
import urllib.request


DEFAULT_MODEL = "llama3.1"
DEFAULT_HOST = "http://localhost:11434"

def load_context_with_langchain(path, chunks):
    try:
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise RuntimeError(
            "LangChain is required for --context-file or --context-dir. Install with:\n"
            "  pip install langchain-community langchain-text-splitters"
        ) from exc

    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    selected = split_docs[: max(chunks, 1)]
    return "\n\n".join(doc.page_content for doc in selected)

def load_context_dir_with_langchain(path, chunks):
    try:
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise RuntimeError(
            "LangChain is required for --context-file or --context-dir. Install with:\n"
            "  pip install langchain-community langchain-text-splitters"
        ) from exc

    loader = DirectoryLoader(
        path,
        glob="**/*",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    selected = split_docs[: max(chunks, 1)]
    return "\n\n".join(doc.page_content for doc in selected)


def stream_chat(host, model, messages):
    url = f"{host}/api/chat"
    payload = json.dumps({"model": model, "messages": messages, "stream": True}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

    full_text = []
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if "message" in data and "content" in data["message"]:
                chunk = data["message"]["content"]
                full_text.append(chunk)
                print(chunk, end="", flush=True)
            if data.get("done"):
                break
        print()
    return "".join(full_text)


def chat_once(host, model, messages):
    url = f"{host}/api/chat"
    payload = json.dumps({"model": model, "messages": messages, "stream": False}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="Chat with a local Ollama model.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama host URL")
    parser.add_argument("--system", default="", help="Optional system prompt")
    parser.add_argument("--context-file", default="", help="Path to a text file for context (LangChain)")
    parser.add_argument("--context-dir", default="", help="Directory for context (LangChain). Defaults to the AI folder.")
    parser.add_argument("--no-context", action="store_true", help="Disable file-based context loading")
    parser.add_argument("--context-chunks", type=int, default=2, help="Number of chunks to include")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    args = parser.parse_args()

    messages = []
    system_parts = []
    if args.system:
        system_parts.append(args.system)
    if not args.no_context:
        try:
            if args.context_file:
                context = load_context_with_langchain(args.context_file, args.context_chunks)
                system_parts.append("Context:\n" + context)
            else:
                if not args.context_dir:
                    args.context_dir = os.path.dirname(__file__)
                if os.path.isdir(args.context_dir):
                    context = load_context_dir_with_langchain(args.context_dir, args.context_chunks)
                    system_parts.append("Context:\n" + context)
        except RuntimeError as exc:
            print(f"Warning: {exc}")
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    print("Type your message. Use /exit to quit.")
    while True:
        try:
            user_text = input("> ").strip()
        except EOFError:
            print()
            break

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            break

        messages.append({"role": "user", "content": user_text})

        try:
            if args.no_stream:
                reply = chat_once(args.host, args.model, messages)
                print(reply)
            else:
                reply = stream_chat(args.host, args.model, messages)
        except Exception as exc:
            print(f"Error talking to Ollama: {exc}")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    sys.exit(main())

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::io;
use std::path::Path;
use async_openai::Client;
use anyhow::{anyhow, Result, Context};
use async_openai::types::{ChatCompletionRequestMessageArgs, CreateChatCompletionRequestArgs, CreateChatCompletionResponse, Role};
use log::{debug, info};
use tiktoken_rust::Encoding;

const CHUNK_SPLIT_SIZE: usize = 2000;
const CONTEXT_SIZE: usize = 50;

pub trait PdfReader {
}

trait PdfWriter {
}

#[derive(Debug, Default)]
pub struct PdfClient {}

impl PdfClient {

}

#[derive(Default)]
pub struct OpenAIClientBuilder {
    api_key: String,
    http_proxy: Option<String>,
    model: String,
    max_token: u16,
    encoding: Option<Encoding>,
}

impl OpenAIClientBuilder {

    pub fn with_api_key(mut self, api_key: String) -> Self {
        self.api_key = api_key;
        self
    }

    pub fn with_http_proxy(mut self, http_proxy: Option<String>) -> Self {
        self.http_proxy = http_proxy;
        self
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    pub fn with_max_token(mut self, max_token: u16) -> Self {
        self.max_token = max_token;
        self
    }

    pub fn with_encoding(mut self, encoding: Encoding) -> Self {
        self.encoding = Some(encoding);
        self
    }

    pub fn build(self) -> Result<OpenAIClient> {
        if self.api_key.is_empty() {
            return Err(anyhow!("OpenAI API KEY is missing"));
        }
        if self.model.is_empty() {
            return Err(anyhow!("AI model name is missing"));
        }
        if self.encoding.is_none() {
            return Err(anyhow!("Encoding is missing"));
        }

        let client = match self.http_proxy {
            Some(uri) => {
                debug!("-- create http client with proxy");
                let proxy = reqwest::Proxy::all(uri)?;
                let http_client = reqwest::Client::builder()
                    .proxy(proxy)
                    .build()?;
                Client::new().with_http_client(http_client)
            }
            None => Client::new(),
        };

        Ok(OpenAIClient{
            client,
            model: self.model,
            max_token: self.max_token,
            encoding: self.encoding.unwrap(),
        })
    }
}

#[derive(Debug)]
pub struct OpenAIClient {
    client: Client,
    model: String,
    max_token: u16,
    encoding: Encoding,
}

impl OpenAIClient {

    pub fn builder() -> OpenAIClientBuilder {
        OpenAIClientBuilder::default()
    }

    pub async fn do_summary(&self, prev: &str, curr: &str, next: &str) -> Result<String> {
        let system_prompt = ChatCompletionRequestMessageArgs::default()
            .role(Role::System)
            .content(r"\
You are an assistant. \
Your only job is summarize a PDF file. File content is provided as text chunks by the user. \
When summarize each chunk, you should consider a few more words from previous chunk \
and a few words from next chunk for better result. \
The following format will be used to send text chunk:\n \
<previous>
`````` // empty OR words from last chunk sent to you, delimited with triple backticks

<current>
`````` // text to summarize this iteration, delimited with triple backticks

<next>
`````` // empty OR words from next chunk will send to you, delimited with triple backticks

The summary should mimic tone of original content, and pay attention to relation between chunks.
organize summary in paragraphs")
            .build()?;

        let content = format!(r"
<previous>
```{}```

<current>
```{}```

<next>
```{}```
", prev, curr, next);
        debug!("chat prompt: {}", content);
        let user_prompt = ChatCompletionRequestMessageArgs::default()
            .role(Role::User)
            .content(content)
            .build()?;

        let request = CreateChatCompletionRequestArgs::default()
            .model(self.model.as_str())
            .temperature(0.2)
            .messages([system_prompt, user_prompt])
            .build()?;

        let resp = self.client
            .chat()
            .create(request)
            .await
            .with_context(|| format!("openai chat failed"))?;

        for choice in resp.choices {
            debug!("-- `send_chunk` reply {}:\nRole: {}\nContent: {}",
                choice.index, choice.message.role, choice.message.content);
            return Ok(choice.message.content)
        }

        Err(anyhow!("no summary found."))
    }

    pub async fn summarization(&self, original: String) -> Result<String> {
        if original.is_empty() {
            return Err(anyhow!("nothing to summarize, just return"))
        }

        let tokens = self.encoding.encode_ordinary(&original);
        let mut chunks = tokens.chunks(CHUNK_SPLIT_SIZE);
        if chunks.len() == 0 {
            return Err(anyhow!("no tokens found for input text"))
        }
        let mut prev_str = String::new();
        let mut next_str = String::new();

        let mut curr_chunk = chunks.next().unwrap();
        let mut curr_str = self.encoding.decode(curr_chunk, tiktoken_rust::DecodeMode::Replace)?;

        let mut res = vec![];
        // chunks
        while chunks.next().is_some() {
            let next_chunk = chunks.next().unwrap();
            let next_chunk_head = &next_chunk[0..CONTEXT_SIZE];
            next_str = self.encoding.decode(next_chunk_head, tiktoken_rust::DecodeMode::Replace)?;

            res.push(self.do_summary(&prev_str, &curr_str, &next_str).await?);

            let curr_chunk_tail = &curr_chunk[curr_chunk.len()-CONTEXT_SIZE..];
            prev_str = self.encoding.decode(curr_chunk_tail, tiktoken_rust::DecodeMode::Replace)?;

            curr_chunk = next_chunk;
            curr_str = self.encoding.decode(curr_chunk, tiktoken_rust::DecodeMode::Replace)?;
        }
        // last chunk
        res.push(self.do_summary(&prev_str, &curr_str, "").await?);

        Ok(res.join("\n"))
    }
}

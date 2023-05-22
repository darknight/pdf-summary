use anyhow::Result;
use clap::Parser;
use env_logger::Env;
use log::{debug, info};
use tiktoken_rust::Encoding;
use tokio::{fs, task};
use pdf_summary::{OpenAIClient, PdfClient};

const DEFAULT_MODEL: &str = "gpt-3.5-turbo";
const TOKEN_LIMIT: u16 = 4000;

#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// the path of PDF file to summarize
    #[arg(short, long)]
    pdf_file: String,
    /// path of generated summary file to write to
    #[arg(short, long)]
    summary_file: String,
}

#[tokio::main]
async fn main() -> Result<()> {

    // load env
    dotenvy::dotenv()?;

    // init logger
    let log_env = Env::default()
        .filter_or("LOG_LEVEL", "debug")
        .write_style_or("LOG_STYLE", "always");
    env_logger::init_from_env(log_env);
    debug!("load env & init logger done...");

    let encoding = task::spawn_blocking(move || {
        tiktoken_rust::encoding_for_model(DEFAULT_MODEL)
    }).await?;
    let openai_client = get_openai_client(encoding?)?;
    debug!("init openai client done...");

    // cmd arguments
    let args = Args::parse();
    let text = pdf_extract::extract_text(args.pdf_file)?;
    // debug!("pdf content: {:?}", text);

    let summary = openai_client.summarization(text).await?;
    save_summary(summary, args.summary_file).await?;

    Ok(())
}

fn get_openai_client(encoding: Encoding) -> Result<OpenAIClient> {
    // init openai client
    let api_key = dotenvy::var("OPENAI_API_KEY")?;
    let proxy = dotenvy::var("HTTP_PROXY")
        .ok()
        .filter(|uri| uri.starts_with("http"));

    OpenAIClient::builder()
        .with_api_key(api_key)
        .with_http_proxy(proxy)
        .with_model(DEFAULT_MODEL)
        .with_max_token(TOKEN_LIMIT)
        .with_encoding(encoding)
        .build()
}

async fn save_summary(summary: String, summary_file: String) -> Result<()> {
    Ok(fs::write(summary_file, summary).await?)
}

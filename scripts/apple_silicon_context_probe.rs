//! Exact-token long-context probe for a running SGLang server.
//!
//! Uses the native `/generate` token-id path, keeping Python and tokenizer
//! work outside the measurement. Build with:
//!
//! ```text
//! rustc +1.92 --edition 2024 -O scripts/apple_silicon_context_probe.rs \
//!   -o /tmp/sglang-context-probe
//! ```

use std::env;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

fn parse_arg<T: std::str::FromStr>(value: Option<String>, name: &str) -> T {
    value
        .unwrap_or_else(|| panic!("missing {}", name))
        .parse()
        .unwrap_or_else(|_| panic!("invalid {}", name))
}

fn main() {
    let mut args = env::args().skip(1);
    let prompt_tokens: usize = parse_arg(args.next(), "prompt_tokens");
    let max_new_tokens: usize = args
        .next()
        .map_or(1, |value| value.parse().expect("invalid max_new_tokens"));
    let address = args.next().unwrap_or_else(|| "127.0.0.1:30000".into());
    assert!(prompt_tokens > 0, "prompt_tokens must be positive");

    let mut body = String::with_capacity(prompt_tokens.saturating_mul(4) + 128);
    body.push_str("{\"input_ids\":[");
    for index in 0..prompt_tokens {
        if index != 0 {
            body.push(',');
        }
        body.push_str("100");
    }
    body.push_str("],\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":");
    body.push_str(&max_new_tokens.to_string());
    body.push_str(",\"ignore_eos\":true}}");

    let mut stream = TcpStream::connect(&address).expect("connect failed");
    stream
        .set_read_timeout(Some(Duration::from_secs(4 * 60 * 60)))
        .expect("set_read_timeout failed");
    stream
        .set_write_timeout(Some(Duration::from_secs(60)))
        .expect("set_write_timeout failed");
    let request = format!(
        "POST /generate HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );

    let started = Instant::now();
    stream.write_all(request.as_bytes()).expect("write failed");
    stream
        .write_all(body.as_bytes())
        .expect("write body failed");
    stream.flush().expect("flush failed");

    let mut response = Vec::new();
    stream.read_to_end(&mut response).expect("read failed");
    let elapsed = started.elapsed();
    let text = String::from_utf8_lossy(&response);
    let (headers, response_body) = text
        .split_once("\r\n\r\n")
        .expect("malformed HTTP response");
    let status = headers.lines().next().unwrap_or("missing status");

    println!("status={status}");
    println!("prompt_tokens={prompt_tokens}");
    println!("max_new_tokens={max_new_tokens}");
    println!("elapsed_seconds={:.6}", elapsed.as_secs_f64());
    println!(
        "prompt_throughput_tok_s={:.6}",
        prompt_tokens as f64 / elapsed.as_secs_f64()
    );
    println!("response={response_body}");
}

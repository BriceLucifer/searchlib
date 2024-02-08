use std::arch::x86_64::*;
use std::path::Path;
use walkdir::WalkDir;
use rusqlite::{params, Connection, Result};
use std::error::Error;
use finalfusion::prelude::*;
use regex::Regex;
use std::fs::File;
use std::io::BufReader;


pub fn wildcard_to_regex(wildcard: &str) -> String {
    let mut regex = String::from("^");
    let mut chars = wildcard.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '*' => {
                // 处理连续的星号 '**'
                if chars.peek() == Some(&'*') {
                    // 消费下一个星号
                    chars.next();
                    regex.push_str(".*"); // 匹配任意数量的目录，注意：这里的实现简化了，具体可能需要调整
                } else {
                    regex.push_str("[^/]*"); // 匹配非斜杠的任意字符序列
                }
            },
            '?' => regex.push_str("[^/]"), // 匹配单个非斜杠字符
            '[' => regex.push('['),
            ']' => regex.push(']'),
            // 转义特殊字符
            '.' | '^' | '$' | '(' | ')' | '|' | '+' | '\\' | '/' => {
                regex.push('\\');
                regex.push(c);
            },
            // 简单处理花括号，这需要更复杂的逻辑来完全实现
            '{' => regex.push_str("(?:"),
            '}' => regex.push(')'),
            ',' => regex.push('|'),
            _ => regex.push(c),
        }
    }

    regex.push('$');
    regex
}

// 计算目录大小 就是叠加文件
pub fn calculate_directory_size<P: AsRef<Path>>(path: P) -> Result<u64> {
    let size = WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter_map(|e| e.metadata().ok())
        .filter(|metadata| metadata.is_file())
        .map(|metadata| metadata.len())
        .sum();
    Ok(size)
}

// 计算相似度 文本
pub fn cosine_similarity_simd(a: Vec<f32>, b: Vec<f32>) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must be of the same length");

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    let chunks_a = a.chunks_exact(4);
    let chunks_b = b.chunks_exact(4);
    let remainder_a = chunks_a.remainder();
    let remainder_b = chunks_b.remainder();

    unsafe {
        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let a_chunk = _mm_loadu_ps(chunk_a.as_ptr());
            let b_chunk = _mm_loadu_ps(chunk_b.as_ptr());

            let dp = _mm_dp_ps(a_chunk, b_chunk, 0xF1);
            dot_product += _mm_cvtss_f32(dp);

            let na = _mm_dp_ps(a_chunk, a_chunk, 0xF1);
            norm_a += _mm_cvtss_f32(na);

            let nb = _mm_dp_ps(b_chunk, b_chunk, 0xF1);
            norm_b += _mm_cvtss_f32(nb);
        }
    }

    for (&a, &b) in remainder_a.iter().zip(remainder_b) {
        dot_product += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    dot_product / (norm_a.sqrt() * norm_b.sqrt())
}


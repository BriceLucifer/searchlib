use bytesize::ByteSize;
use clap::{Args, Parser, Subcommand};
use colored::Colorize;
use finalfusion::prelude::*;
use regex::Regex;
use rusqlite::{params, Connection, Result};
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use walkdir::WalkDir;

mod features;
use features::{calculate_directory_size, cosine_similarity_simd, wildcard_to_regex};

#[derive(Parser)]
#[command(
    author = "fb8py",
    version = "1.0",
    about = "Directory and file processing utility."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initializes the target directory
    Init(Initpath),
    /// 正则表达式匹配
    Search(Filename),
    ///丢你老母
    Delaynomore(Filename),
}

#[derive(Args)]
struct Initpath {
    path: String,
}

#[derive(Args)]
struct Filename {
    name: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let mut conn = Connection::open("./temp.db")?;

    let mut reader = BufReader::new(File::open("./src/fasttext.bin").unwrap());
    let embeds = Embeddings::read_fasttext(&mut reader).unwrap();

    conn.execute_batch(
        "
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA cache_size = 10000;
        PRAGMA locking_mode = EXCLUSIVE;
        PRAGMA temp_store = MEMORY;
        ",
    )?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            path TEXT,
            type TEXT,
            size INTEGER
        );",
        [],
    )?;

    match &cli.command {
        Commands::Init(initpath) => {
            let tx = conn.transaction()?;
            for entry in WalkDir::new(&initpath.path)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let metadata = entry.metadata().unwrap();

                let file_name = entry.file_name().to_string_lossy().to_string();
                let path = entry.path().to_string_lossy().to_string();

                let item_type = if metadata.is_dir() {
                    "文件夹".to_string()
                } else {
                    entry
                        .path()
                        .extension()
                        .map(|ext| ext.to_string_lossy().to_string())
                        .unwrap_or("文件".to_string())
                };

                let size = if metadata.is_dir() {
                    calculate_directory_size(entry.path())?
                } else {
                    metadata.len()
                };

                tx.execute(
                    "INSERT OR IGNORE INTO items (name, path, type, size) VALUES (?1, ?2, ?3, ?4)",
                    params![file_name, path, item_type, size as i64],
                )?;

                let formatted_entry = if metadata.is_dir() {
                    path.blue()
                } else {
                    path.yellow()
                };
                println!(
                    "{:>9}\t{}",
                    ByteSize(size).to_string().green(),
                    formatted_entry
                );
            }
            tx.commit()?;
        }

        Commands::Search(filename) => {
            // 转换通配符为正则表达式
            let pattern = wildcard_to_regex(&filename.name);
            let regex = Regex::new(&pattern).expect("Invalid regex pattern");

            // 准备 SQL 查询，这次我们不在 SQL 里使用 pattern，因为正则匹配在 Rust 中完成
            let mut stmt = conn.prepare("SELECT name, path, size FROM items")?;
            // 执行查询，不需要传递 pattern 作为参数
            let rows = stmt.query_map(params![], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, u64>(2)?,
                ))
            })?;

            // 遍历查询结果，并使用 Rust 的正则表达式库进行匹配
            for row in rows {
                let (name, path, size) = row?;
                // 这里用 regex.is_match 检查 name 是否匹配用户给定的模式
                if regex.is_match(&name) {
                    // 如果匹配，打印信息
                    println!(
                        "{}: {} ({})",
                        name.yellow(),
                        path.blue(),
                        ByteSize(size).to_string().green()
                    );
                }
            }
        }
        Commands::Delaynomore(filename) => {
            let conn_temp = Connection::open_in_memory()?;
            conn_temp.execute(
                "CREATE TABLE memory_temp (
                    id   INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    path TEXT,
                    similarity NUMERIC);",
                [],
            )?;

            let mut stmt = conn.prepare(
                "SELECT name,path
                        FROM items
                        WHERE type 
                        NOT IN ('文件夹');",
            )?;
            let rows = stmt.query_map(params![], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            for row in rows {
                let (name, path) = row?;
                let embds_result_1 = embeds.embedding(&name).unwrap().to_vec();
                // embds_result_2是用户输入的参数
                let embds_result_2 = embeds
                    .embedding(&filename.name.to_string())
                    .unwrap()
                    .to_vec();
                let embds_result_3 = cosine_similarity_simd(embds_result_1, embds_result_2);

                conn_temp.execute(
                    "INSERT INTO memory_temp (name,path,similarity) VALUES (?1,?2,?3);",
                    params![name, path, embds_result_3],
                )?;
            }
            // print!("到这里了！");
            let mut stmt1 = conn_temp.prepare(
                "SELECT id,name,path,similarity 
                        FROM memory_temp 
                        ORDER BY similarity 
                        DESC LIMIT 0,5;",
            )?;
            let rows = stmt1.query_map(params![], |row| {
                Ok((
                    row.get::<_, u32>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, f32>(3)?,
                ))
            })?;
            for row in rows {
                let (id, name, path, similarity) = row?;
                println!(
                    "{}: {} {} {};",
                    id.to_string().white(),
                    name.yellow(),
                    path.blue(),
                    similarity.to_string().green(),
                );
            }
        }
    }
    Ok(())
}

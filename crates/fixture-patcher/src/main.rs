//! Identifies and reports execution witness fixtures missing the state trie root node.
//!
//! Reth's `debug_executionWitness` omits the root node for historical blocks.
//! This tool scans all fixtures, reports which ones are valid vs broken,
//! and writes a CSV summary.

use alloy_consensus::Header;
use alloy_primitives::keccak256;
use alloy_rlp::Decodable;
use anyhow::{Context, Result};
use clap::Parser;
use std::{io::Write, path::PathBuf};
use tracing::{info, warn};
use walkdir::WalkDir;
use witness_generator::StatelessValidationFixture;

#[derive(Parser)]
#[command(about = "Scan witness fixtures for missing state trie root nodes")]
struct Args {
    /// Directory of fixture JSON files to scan
    #[arg(long)]
    fixtures_dir: PathBuf,

    /// Output CSV path
    #[arg(long, default_value = "fixture_scan.csv")]
    output: PathBuf,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();

    let mut files: Vec<PathBuf> = Vec::new();
    for entry in WalkDir::new(&args.fixtures_dir)
        .min_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.path().extension().is_some_and(|ext| ext == "json") {
            files.push(entry.path().to_path_buf());
        }
    }
    files.sort();
    info!("Found {} fixture files", files.len());

    let mut out = std::fs::File::create(&args.output)?;
    writeln!(
        out,
        "filename,block_number,has_root,state_nodes,parent_state_root"
    )?;

    let mut ok = 0usize;
    let mut missing = 0usize;
    let mut errs = 0usize;

    for (i, fpath) in files.iter().enumerate() {
        match check_fixture(fpath) {
            Ok((block_num, has_root, n_nodes, sr)) => {
                let fname = fpath.file_name().unwrap().to_string_lossy();
                writeln!(out, "{fname},{block_num},{has_root},{n_nodes},{sr:?}")?;
                if has_root {
                    ok += 1;
                } else {
                    missing += 1;
                }
            }
            Err(e) => {
                warn!("{}: {e:#}", fpath.display());
                errs += 1;
            }
        }
        if (i + 1) % 100 == 0 || i + 1 == files.len() {
            info!(
                "[{}/{}] root_ok={ok} root_missing={missing} errors={errs}",
                i + 1,
                files.len()
            );
        }
    }

    info!("Wrote {}", args.output.display());
    info!("Summary: {ok} valid, {missing} missing root, {errs} errors");
    if missing > 0 {
        info!(
            "Fixtures with missing root nodes cannot be used for stateless validation. \
             See https://github.com/paradigmxyz/reth/issues/XXXX"
        );
    }

    Ok(())
}

fn check_fixture(fpath: &PathBuf) -> Result<(u64, bool, usize, alloy_primitives::B256)> {
    let raw = std::fs::read(fpath).context("read")?;
    let fixture: StatelessValidationFixture =
        serde_json::from_slice(&raw).context("parse JSON")?;

    let ancestor_bytes = fixture
        .stateless_input
        .witness
        .headers
        .first()
        .context("no ancestor headers")?;
    let ancestor = Header::decode(&mut &ancestor_bytes[..]).context("decode header")?;
    let pre_state_root = ancestor.state_root;

    let n_nodes = fixture.stateless_input.witness.state.len();

    let has_root = fixture
        .stateless_input
        .witness
        .state
        .iter()
        .any(|node| keccak256(node) == pre_state_root);

    let block_num = fixture.stateless_input.block.header.number;

    Ok((block_num, has_root, n_nodes, pre_state_root))
}

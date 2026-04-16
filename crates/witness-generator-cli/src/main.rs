//! CLI to generate witness fixtures with per-block timing metrics.

#![cfg_attr(not(test), warn(unused_crate_dependencies))]

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use std::{io::Write, path::PathBuf, time::Instant};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;
use witness_generator::{
    Fixture, FixtureGenerator, StatelessValidationFixture,
    eest_generator::EESTFixtureGeneratorBuilder,
    raw_input_generator::RawInputFixtureGeneratorBuilder,
    rpc_generator::{RpcBlocksAndWitnessesBuilder, RpcFlatHeaderKeyValues},
};

// --- direct RPC imports for per-block timing ---
use alloy_eips::BlockNumberOrTag;
use alloy_rpc_types_eth::{Block, Header, Receipt, Transaction, TransactionRequest};
use jsonrpsee::http_client::{HeaderMap, HttpClient, HttpClientBuilder};
use reth_chainspec::{Chain, HOLESKY, HOODI, NamedChain, SEPOLIA, mainnet_chain_config};
use reth_ethereum_primitives::TransactionSigned;
use reth_rpc_api::{DebugApiClient, EthApiClient};
use stateless::StatelessInput;

#[derive(Parser)]
#[command(name = "zkvm-fixture-generator")]
#[command(about = "Generate witness fixtures with per-block timing metrics")]
#[command(version)]
struct Cli {
    /// Output folder for generated fixtures
    #[arg(short, long, default_value = "zkevm-fixtures-input")]
    output_folder: PathBuf,

    /// Path for metrics CSV (per-block witness generation timing + metadata)
    #[arg(long)]
    metrics_csv: Option<PathBuf>,

    /// Skip saving fixture JSON files (metrics-only mode)
    #[arg(long)]
    no_save: bool,

    /// Source of blocks and witnesses
    #[command(subcommand)]
    source: SourceCommand,
}

#[derive(Subcommand, Clone, Debug)]
enum SourceCommand {
    /// Generate fixtures from execution specification tests
    Tests {
        #[arg(short, long, conflicts_with = "eest_fixtures_path")]
        tag: Option<String>,
        #[arg(short, long)]
        include: Option<Vec<String>>,
        #[arg(short, long)]
        exclude: Option<Vec<String>>,
        #[arg(long, conflicts_with = "tag")]
        eest_fixtures_path: Option<PathBuf>,
    },
    /// Generate fixtures from raw stateless input URLs
    RawInput {
        #[arg(long)]
        input_folder: PathBuf,
    },
    /// Generate fixtures from an RPC endpoint (with per-block timing)
    Rpc {
        #[arg(long, conflicts_with_all = ["block", "follow"])]
        last_n_blocks: Option<usize>,
        #[arg(long, conflicts_with_all = ["last_n_blocks", "follow"])]
        block: Option<u64>,
        #[arg(long, default_value_t = false, conflicts_with_all = ["last_n_blocks", "block"])]
        follow: bool,
        #[arg(long)]
        rpc_url: String,
        #[arg(long)]
        rpc_header: Option<Vec<String>>,
    },
}

fn classify(n_state: usize) -> &'static str {
    if n_state < 8_000 {
        "light"
    } else if n_state < 12_000 {
        "moderate"
    } else if n_state < 18_000 {
        "heavy"
    } else {
        "extreme"
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();

    // For RPC source with metrics, use our custom per-block timed pipeline
    if let SourceCommand::Rpc {
        ref rpc_url,
        ref rpc_header,
        last_n_blocks,
        block,
        follow,
    } = cli.source
    {
        if !follow {
            return run_rpc_with_timing(
                rpc_url,
                rpc_header.as_deref(),
                last_n_blocks,
                block,
                &cli.output_folder,
                cli.metrics_csv.as_ref(),
                cli.no_save,
            )
            .await;
        }
    }

    // Fallback: use the original generator for non-RPC sources / follow mode
    if !cli.no_save {
        if !cli.output_folder.exists() {
            std::fs::create_dir_all(&cli.output_folder)?;
        }
    }
    let count = build_generator(cli.source)
        .await?
        .generate_to_path(&cli.output_folder)
        .await
        .context("Failed to generate")?;
    info!("Generated {count} fixtures");
    Ok(())
}

/// RPC pipeline with per-block timing — the main addition.
async fn run_rpc_with_timing(
    rpc_url: &str,
    rpc_headers: Option<&[String]>,
    last_n_blocks: Option<usize>,
    block_num: Option<u64>,
    output_folder: &PathBuf,
    metrics_csv: Option<&PathBuf>,
    no_save: bool,
) -> Result<()> {
    // Build HTTP client
    let mut header_map = HeaderMap::new();
    if let Some(hdrs) = rpc_headers {
        for h in hdrs {
            let (k, v) = h
                .split_once(':')
                .ok_or_else(|| anyhow!("bad header: {h}"))?;
            header_map.insert(
                k.trim().parse::<http::HeaderName>()?,
                v.trim().parse::<http::HeaderValue>()?,
            );
        }
    }
    let client = HttpClientBuilder::default()
        .set_headers(header_map)
        .max_response_size(1 << 30)
        .request_timeout(std::time::Duration::from_secs(600))
        .build(rpc_url)?;

    // Get chain config
    let chain_id = EthApiClient::<(), (), (), (), (), ()>::chain_id(&client)
        .await?
        .ok_or_else(|| anyhow!("no chain_id"))?;
    let chain_config = match Chain::from_id(chain_id.to()).named() {
        Some(NamedChain::Mainnet) => mainnet_chain_config(),
        Some(NamedChain::Sepolia) => SEPOLIA.genesis.config.clone(),
        Some(NamedChain::Hoodi) => HOODI.genesis.config.clone(),
        Some(NamedChain::Holesky) => HOLESKY.genesis.config.clone(),
        _ => return Err(anyhow!("unsupported chain {chain_id}")),
    };

    // Determine block range
    let latest = EthApiClient::<
        TransactionRequest,
        Transaction,
        Block,
        Receipt,
        Header,
        TransactionSigned,
    >::block_by_number(&client, BlockNumberOrTag::Latest, false)
    .await?
    .ok_or_else(|| anyhow!("no latest block"))?;

    let blocks: Vec<u64> = if let Some(n) = last_n_blocks {
        let start = latest.header.number.saturating_sub(n as u64 - 1);
        (start..=latest.header.number).rev().collect()
    } else if let Some(b) = block_num {
        vec![b]
    } else {
        vec![latest.header.number]
    };

    info!(
        "Processing {} blocks ({} → {})",
        blocks.len(),
        blocks.last().unwrap_or(&0),
        blocks.first().unwrap_or(&0)
    );

    // Setup output
    if !no_save && !output_folder.exists() {
        std::fs::create_dir_all(output_folder)?;
    }

    let mut csv = if let Some(path) = metrics_csv {
        let exists = path.exists() && path.metadata().map_or(false, |m| m.len() > 0);
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let mut f = f;
        if !exists {
            writeln!(
                f,
                "block_number,timestamp,txn_count,gas_used,gas_limit,gas_utilization_pct,\
                 witness_time_sec,fixture_size_bytes,witness_state_nodes,witness_codes_count,\
                 witness_keys_count,witness_headers_count,category,fixture_file"
            )?;
        }
        Some(f)
    } else {
        None
    };

    let total_start = Instant::now();
    let mut ok = 0usize;
    let mut errors = 0usize;

    let mut skipped = 0usize;

    for (i, &bn) in blocks.iter().enumerate() {
        // Skip blocks whose fixture already exists on disk
        if !no_save {
            let fixture_path = output_folder.join(format!("rpc_block_{bn}.json"));
            if fixture_path.exists() {
                skipped += 1;
                if skipped <= 5 || skipped % 50 == 0 {
                    info!(
                        "[{}/{}] block {bn}: skipping (fixture exists)",
                        i + 1,
                        blocks.len()
                    );
                }
                continue;
            }
        }

        match process_one_block(&client, &chain_config, bn, output_folder, &mut csv, no_save).await
        {
            Ok(()) => ok += 1,
            Err(e) => {
                warn!("[{}/{}] block {bn}: {e:#}", i + 1, blocks.len());
                errors += 1;
            }
        }
        if (i + 1) % 10 == 0 || i + 1 == blocks.len() {
            let elapsed = total_start.elapsed().as_secs_f64();
            let processed = ok + errors;
            let rate = if elapsed > 0.0 {
                processed as f64 / elapsed
            } else {
                0.0
            };
            let remaining = blocks.len() - i - 1;
            let eta = if rate > 0.0 {
                remaining as f64 / rate
            } else {
                0.0
            };
            info!(
                "[{}/{}] ok={ok} err={errors} skipped={skipped} elapsed={elapsed:.0}s eta={eta:.0}s ({rate:.2} blk/s)",
                i + 1,
                blocks.len()
            );
        }
    }

    info!(
        "Done: {ok} ok, {errors} errors, {skipped} skipped in {:.1}s",
        total_start.elapsed().as_secs_f64()
    );
    if let Some(path) = metrics_csv {
        info!("Metrics: {path:?}");
    }
    Ok(())
}

/// Fetch block + witness, time the witness call, save fixture, write CSV row.
async fn process_one_block(
    client: &HttpClient,
    chain_config: &alloy_genesis::ChainConfig,
    block_num: u64,
    output_folder: &PathBuf,
    csv: &mut Option<std::fs::File>,
    no_save: bool,
) -> Result<()> {
    // Fetch block
    let block = EthApiClient::<
        TransactionRequest,
        Transaction,
        Block<TransactionSigned>,
        Receipt,
        Header,
        TransactionSigned,
    >::block_by_number(client, BlockNumberOrTag::Number(block_num), true)
    .await?
    .ok_or_else(|| anyhow!("block {block_num} not found"))?;

    let txn_count = block.transactions.len();
    let gas_used = block.header.gas_used;
    let gas_limit = block.header.gas_limit;
    let timestamp = block.header.timestamp;
    let gas_pct = if gas_limit > 0 {
        gas_used as f64 / gas_limit as f64 * 100.0
    } else {
        0.0
    };

    // Time the witness generation RPC call
    let t_witness = Instant::now();
    let witness =
        DebugApiClient::<()>::debug_execution_witness(client, BlockNumberOrTag::Number(block_num))
            .await?;
    let witness_time = t_witness.elapsed().as_secs_f64();

    let n_state = witness.state.len();
    let n_codes = witness.codes.len();
    let n_keys = witness.keys.len();
    let n_headers = witness.headers.len();
    let category = classify(n_state);

    // Build fixture
    let fixture = StatelessValidationFixture {
        name: format!("rpc_block_{block_num}"),
        stateless_input: StatelessInput {
            block: block.into_consensus(),
            witness,
            chain_config: chain_config.clone(),
        },
        success: true,
    };

    // Serialize
    let mut buf = Vec::new();
    serde_json::to_writer(&mut buf, &fixture)?;
    let fixture_size = buf.len();

    // Save
    let fixture_file = format!("rpc_block_{block_num}.json");
    if !no_save {
        let path = output_folder.join(&fixture_file);
        std::fs::write(&path, &buf)?;
    }

    // CSV row
    if let Some(f) = csv.as_mut() {
        writeln!(
            f,
            "{block_num},{timestamp},{txn_count},{gas_used},{gas_limit},{gas_pct:.2},\
             {witness_time:.4},{fixture_size},{n_state},{n_codes},{n_keys},{n_headers},\
             {category},{fixture_file}"
        )?;
        f.flush()?;
    }

    info!(
        "block {block_num}: {txn_count} txns, {gas_used} gas ({gas_pct:.0}%), \
         witness={witness_time:.3}s ({fixture_size} bytes, {n_state} state, {n_codes} codes) [{category}]"
    );

    Ok(())
}

async fn build_generator(source: SourceCommand) -> Result<Box<dyn FixtureGenerator>> {
    match source {
        SourceCommand::Tests {
            tag,
            include,
            exclude,
            eest_fixtures_path,
        } => {
            let mut builder = EESTFixtureGeneratorBuilder::default();
            if let Some(tag) = tag {
                builder = builder.with_tag(tag);
            } else if let Some(p) = eest_fixtures_path {
                builder = builder.with_input_folder(p)?;
            }
            if let Some(inc) = include {
                builder = builder.with_includes(inc);
            }
            if let Some(exc) = exclude {
                builder = builder.with_excludes(exc);
            }
            Ok(Box::new(builder.build().await?))
        }
        SourceCommand::RawInput { input_folder } => Ok(Box::new(
            RawInputFixtureGeneratorBuilder::default()
                .with_input_folder(input_folder)?
                .build()?,
        )),
        SourceCommand::Rpc {
            rpc_url,
            rpc_header,
            follow,
            last_n_blocks,
            block,
        } => {
            let mut builder = RpcBlocksAndWitnessesBuilder::new(rpc_url);
            if let Some(hdrs) = rpc_header {
                builder = builder.with_headers(RpcFlatHeaderKeyValues::new(hdrs).try_into()?);
            }
            if follow {
                let stop = CancellationToken::new();
                builder = builder.listen(stop.clone());
                tokio::spawn(async move {
                    let _ = tokio::signal::ctrl_c().await;
                    info!("Stopping...");
                    stop.cancel();
                });
            } else if let Some(b) = block {
                builder = builder.block(b);
            } else {
                builder = builder.last_n_blocks(last_n_blocks.unwrap_or(1));
            }
            Ok(Box::new(builder.build().await?))
        }
    }
}

//! CLI to generate witness fixtures for zkEVM benchmarking, with optional per-block timing
//! metrics and a resumable RPC pipeline.

#![cfg_attr(not(test), warn(unused_crate_dependencies))]

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use std::{io::Write, path::PathBuf, time::Instant};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;
use witness_generator::{
    FixtureGenerator, StatelessValidationFixture,
    eest_generator::EESTFixtureGeneratorBuilder,
    raw_input_generator::RawInputFixtureGeneratorBuilder,
    rpc_generator::{RpcBlocksAndWitnessesBuilder, RpcFlatHeaderKeyValues},
};

use alloy_eips::BlockNumberOrTag;
use alloy_genesis::{ChainConfig, Genesis};
use alloy_rpc_types_eth::{Block, Header, Receipt, Transaction, TransactionRequest};
use jsonrpsee::http_client::{HeaderMap, HttpClient, HttpClientBuilder};
use reth_chainspec::{Chain, HOLESKY, HOODI, NamedChain, SEPOLIA, mainnet_chain_config};
use reth_ethereum_primitives::TransactionSigned;
use reth_rpc_api::{DebugApiClient, EthApiClient};
use stateless::StatelessInput;

#[derive(Parser)]
#[command(name = "zkvm-fixture-generator")]
#[command(about = "Generate fixtures for zkEVM benchmarking tool")]
#[command(version)]
struct Cli {
    /// Output folder for generated fixtures
    #[arg(short, long, default_value = "zkevm-fixtures-input")]
    output_folder: PathBuf,

    /// Optional CSV path for per-block witness-generation timing metrics (RPC source only).
    /// Appends if the file already exists; writes a header on first creation.
    #[arg(long)]
    metrics_csv: Option<PathBuf>,

    /// Skip writing fixture JSON files. Useful with `--metrics-csv` for metrics-only runs.
    #[arg(long, default_value_t = false)]
    no_save: bool,

    /// Source of blocks and witnesses
    #[command(subcommand)]
    source: SourceCommand,
}

#[derive(Subcommand, Clone, Debug)]
enum SourceCommand {
    /// Generate fixtures from execution specification tests
    Tests {
        /// EEST release tag to use (e.g., "v0.1.0"). If empty, the latest release will be used.
        #[arg(short, long, conflicts_with = "eest_fixtures_path")]
        tag: Option<String>,

        /// Include only test names containing the provided strings.
        #[arg(short, long)]
        include: Option<Vec<String>>,

        /// Exclude all test names containing the provided strings.
        #[arg(short, long)]
        exclude: Option<Vec<String>>,

        /// Optional input folder for EEST files. If not provided, the tag rule will be used.
        #[arg(long, conflicts_with = "tag")]
        eest_fixtures_path: Option<PathBuf>,
    },
    /// Generate fixtures from raw stateless input URLs listed in `raw_input_parts.txt`
    RawInput {
        /// Path to the input folder containing `chain_config.json` and `raw_input_parts.txt`
        #[arg(long)]
        input_folder: PathBuf,
    },
    /// Generate fixtures from an RPC endpoint
    Rpc {
        /// Number of last blocks to pull
        #[arg(long, conflicts_with_all = ["block", "follow"])]
        last_n_blocks: Option<usize>,

        /// Specific block number to pull
        #[arg(long, conflicts_with_all = ["last_n_blocks", "follow"])]
        block: Option<u64>,

        /// Listen for new blocks
        #[arg(long, default_value_t = false, conflicts_with_all = ["last_n_blocks", "block"])]
        follow: bool,

        /// RPC URL to use (mandatory)
        #[arg(long)]
        rpc_url: String,

        /// Optional RPC headers to use (format: "Key:Value")
        #[arg(long)]
        rpc_header: Option<Vec<String>>,

        /// Optional path to a geth-style genesis.json file for custom/devnet chain config
        #[arg(long, value_name = "PATH")]
        genesis: Option<PathBuf>,
    },
}

/// Categorize a block by its witness state-trie node count. Boundaries are derived from
/// observed mainnet percentiles and are useful for downstream benchmark grouping.
const fn classify(n_state: usize) -> &'static str {
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

    // Per-block timing pipeline: only meaningful for one-shot RPC runs (not `--follow`).
    if let SourceCommand::Rpc {
        ref rpc_url,
        ref rpc_header,
        ref genesis,
        last_n_blocks,
        block,
        follow,
    } = cli.source
        && !follow
    {
        return run_rpc_with_timing(
            rpc_url,
            rpc_header.as_deref(),
            genesis.as_deref(),
            last_n_blocks,
            block,
            &cli.output_folder,
            cli.metrics_csv.as_deref(),
            cli.no_save,
        )
        .await;
    }

    info!("Generating fixtures in folder: {:?}", cli.output_folder);
    if !cli.no_save && !cli.output_folder.exists() {
        std::fs::create_dir_all(&cli.output_folder)
            .with_context(|| format!("Failed to create output folder: {:?}", cli.output_folder))?;
    }

    info!("Generating fixtures...");
    let count = build_generator(cli.source)
        .await?
        .generate_to_path(&cli.output_folder)
        .await
        .context("Failed to generate blocks and witnesses")?;

    info!("Generated {} blocks and witnesses", count);

    Ok(())
}

/// One-shot RPC pipeline that times the `debug_executionWitness` call per block,
/// optionally appends a CSV row of metrics, and skips blocks whose fixture already
/// exists on disk so runs are resumable.
#[allow(clippy::too_many_arguments)]
async fn run_rpc_with_timing(
    rpc_url: &str,
    rpc_headers: Option<&[String]>,
    genesis_path: Option<&std::path::Path>,
    last_n_blocks: Option<usize>,
    block_num: Option<u64>,
    output_folder: &std::path::Path,
    metrics_csv: Option<&std::path::Path>,
    no_save: bool,
) -> Result<()> {
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

    let chain_id = EthApiClient::<(), (), (), (), (), ()>::chain_id(&client)
        .await?
        .ok_or_else(|| anyhow!("no chain_id"))?;
    let rpc_chain_id: u64 = chain_id.to();

    let chain_config = if let Some(path) = genesis_path {
        let cfg = load_chain_config_from_genesis(path)?;
        if cfg.chain_id != rpc_chain_id {
            return Err(anyhow!(
                "genesis chain ID mismatch for {}: genesis={}, rpc={}",
                path.display(),
                cfg.chain_id,
                rpc_chain_id
            ));
        }
        cfg
    } else {
        match Chain::from_id(rpc_chain_id).named() {
            Some(NamedChain::Mainnet) => mainnet_chain_config(),
            Some(NamedChain::Sepolia) => SEPOLIA.genesis.config.clone(),
            Some(NamedChain::Hoodi) => HOODI.genesis.config.clone(),
            Some(NamedChain::Holesky) => HOLESKY.genesis.config.clone(),
            _ => return Err(anyhow!("unsupported chain {rpc_chain_id}")),
        }
    };

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
        if n == 0 {
            return Err(anyhow!("--last-n-blocks must be greater than 0"));
        }
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

    if !no_save && !output_folder.exists() {
        std::fs::create_dir_all(output_folder)?;
    }

    let mut csv = if let Some(path) = metrics_csv {
        let exists = path.exists() && path.metadata().map(|m| m.len() > 0).unwrap_or(false);
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
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
        if !no_save {
            let fixture_path = output_folder.join(format!("rpc_block_{bn}.json"));
            if fixture_path.exists() {
                skipped += 1;
                if skipped <= 5 || skipped.is_multiple_of(50) {
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

fn load_chain_config_from_genesis(path: &std::path::Path) -> Result<ChainConfig> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read genesis file at {}", path.display()))?;
    let genesis: Genesis = serde_json::from_str(&contents)
        .with_context(|| format!("failed to deserialize genesis file at {}", path.display()))?;
    Ok(genesis.config)
}

/// Fetch block + witness, time the witness call, save fixture, and write a CSV row.
async fn process_one_block(
    client: &HttpClient,
    chain_config: &ChainConfig,
    block_num: u64,
    output_folder: &std::path::Path,
    csv: &mut Option<std::fs::File>,
    no_save: bool,
) -> Result<()> {
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

    let t_witness = Instant::now();
    let witness = DebugApiClient::<()>::debug_execution_witness(
        client,
        BlockNumberOrTag::Number(block_num),
        None,
    )
    .await?;
    let witness_time = t_witness.elapsed().as_secs_f64();

    let n_state = witness.state.len();
    let n_codes = witness.codes.len();
    let n_keys = witness.keys.len();
    let n_headers = witness.headers.len();
    let category = classify(n_state);

    let fixture = StatelessValidationFixture {
        name: format!("rpc_block_{block_num}"),
        stateless_input: StatelessInput {
            block: block.into_consensus(),
            witness,
            chain_config: chain_config.clone(),
        },
        success: true,
    };

    let mut buf = Vec::new();
    serde_json::to_writer(&mut buf, &fixture)?;
    let fixture_size = buf.len();

    let fixture_file = format!("rpc_block_{block_num}.json");
    if !no_save {
        let path = output_folder.join(&fixture_file);
        std::fs::write(&path, &buf)?;
    }

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
            } else if let Some(input_folder) = eest_fixtures_path {
                builder = builder.with_input_folder(input_folder)?;
            }

            if let Some(include) = include {
                builder = builder.with_includes(include);
            }
            if let Some(exclude) = exclude {
                builder = builder.with_excludes(exclude);
            }

            Ok(Box::new(
                builder
                    .build()
                    .await
                    .context("Failed to build EEST generator")?,
            ))
        }
        SourceCommand::RawInput { input_folder } => Ok(Box::new(
            RawInputFixtureGeneratorBuilder::default()
                .with_input_folder(input_folder)
                .context("Invalid raw input folder")?
                .build()
                .context("Failed to build raw input generator")?,
        )),
        SourceCommand::Rpc {
            last_n_blocks,
            block,
            rpc_url,
            rpc_header,
            genesis,
            follow: listen,
        } => {
            let mut builder = RpcBlocksAndWitnessesBuilder::new(rpc_url);

            if let Some(rpc_header) = rpc_header {
                let headers = RpcFlatHeaderKeyValues::new(rpc_header)
                    .try_into()
                    .context("Failed to parse RPC headers")?;
                builder = builder.with_headers(headers);
            }

            if let Some(genesis) = genesis {
                builder = builder.with_genesis(genesis);
            }

            if listen {
                let stop = CancellationToken::new();
                builder = builder.listen(stop.clone());

                tokio::spawn(async move {
                    tokio::select! {
                        _ = tokio::signal::ctrl_c() => {
                            info!("Stopping...");
                            stop.cancel();
                        }
                    }
                });
            } else if let Some(block_num) = block {
                builder = builder.block(block_num);
            } else {
                let n_blocks = last_n_blocks.unwrap_or(1);
                if n_blocks == 0 {
                    return Err(anyhow!("Number of blocks must be greater than 0"));
                }
                builder = builder.last_n_blocks(n_blocks);
            }

            Ok(Box::new(
                builder
                    .build()
                    .await
                    .context("Failed to build RPC generator")?,
            ))
        }
    }
}

//! BuilderTee HTTP server — real Reth stateless block validation.
//!
//! Accepts execution witness + block fixtures via HTTP, runs the full
//! Reth stateless validation pipeline (MPT proof verification, EVM
//! execution, post-state root check), and returns per-stage wall-clock
//! timing in the response.
//!
//! # Endpoints
//!
//! - `POST /witness`       — full fixture JSON body; deserialise + validate
//! - `POST /validate/{idx}` — validate pre-loaded fixture by index (no deser)
//! - `GET  /info`          — fixture count + aggregate stats
//! - `GET  /health`        — liveness probe

#![cfg_attr(not(test), warn(unused_crate_dependencies))]

use anyhow::{Context, Result};
use axum::{
    Router,
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
};
use clap::Parser;
use ere_guests_guest::Guest;
use ere_guests_integration_tests::NoopPlatform;
use ere_guests_stateless_validator_reth::guest::{
    StatelessValidatorRethGuest, StatelessValidatorRethInput,
};
use serde::Serialize;
use std::{net::SocketAddr, sync::Arc, time::Instant};
use tracing::{info, warn};
use walkdir::WalkDir;
use witness_generator::StatelessValidationFixture;

// ── CLI ──────────────────────────────────────────────────────────────

/// BuilderTee HTTP server with real Reth stateless validation.
#[derive(Parser)]
#[command(about)]
struct Args {
    /// Socket address to listen on.
    #[arg(long, default_value = "127.0.0.1:50090")]
    listen: SocketAddr,

    /// Directory of `.json` fixture files to pre-load (for `/validate/{idx}`).
    #[arg(long)]
    fixtures_dir: std::path::PathBuf,

    /// Skip fixtures larger than this many bytes (0 = no limit).
    #[arg(long, default_value_t = 50_000_000)]
    max_fixture_bytes: usize,
}

// ── Shared state ─────────────────────────────────────────────────────

struct AppState {
    fixtures: Vec<LoadedFixture>,
}

struct LoadedFixture {
    fixture: StatelessValidationFixture,
    raw_size: usize,
    /// Number of state trie nodes in the witness.
    n_state_nodes: usize,
    /// Number of unique code entries in the witness.
    n_codes: usize,
    /// Number of transactions in the block body.
    n_txs: usize,
    /// Complexity category derived from witness characteristics.
    category: &'static str,
}

/// Classify a fixture as lightweight / moderate / heavy / worst-case based on
/// witness size, state node count, and transaction count.
fn classify_fixture(raw_size: usize, n_state: usize, n_txs: usize) -> &'static str {
    if n_state <= 20 && n_txs <= 2 && raw_size < 100_000 {
        "lightweight"
    } else if n_state <= 200 && n_txs <= 50 {
        "moderate"
    } else if n_state <= 2000 {
        "heavy"
    } else {
        "worst_case"
    }
}

// ── Response types ───────────────────────────────────────────────────

#[derive(Serialize)]
struct ValidationResponse {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    fixture_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fixture_idx: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    witness_size_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    block_gas_used: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n_state_nodes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n_txs: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    category: Option<String>,
    /// Time to deserialise the fixture JSON (ms). Only for `/witness`.
    #[serde(skip_serializing_if = "Option::is_none")]
    deser_ms: Option<f64>,
    /// Time to create `StatelessValidatorRethInput` from `StatelessInput` (ms).
    #[serde(skip_serializing_if = "Option::is_none")]
    input_prep_ms: Option<f64>,
    /// Time for full Reth stateless validation: witness verify + EVM exec +
    /// post-state root check (ms).
    #[serde(skip_serializing_if = "Option::is_none")]
    validation_ms: Option<f64>,
    /// Total server-side wall-clock time (ms).
    #[serde(skip_serializing_if = "Option::is_none")]
    total_server_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl Default for ValidationResponse {
    fn default() -> Self {
        Self {
            status: "error".into(),
            fixture_name: None,
            fixture_idx: None,
            witness_size_bytes: None,
            block_gas_used: None,
            n_state_nodes: None,
            n_txs: None,
            category: None,
            deser_ms: None,
            input_prep_ms: None,
            validation_ms: None,
            total_server_ms: None,
            error: None,
        }
    }
}

#[derive(Serialize)]
struct CategoryStats {
    count: usize,
    avg_size_kb: f64,
    indices: Vec<usize>,
}

#[derive(Serialize)]
struct InfoResponse {
    fixture_count: usize,
    total_size_mb: f64,
    avg_size_kb: f64,
    categories: std::collections::HashMap<String, CategoryStats>,
}

// ── Fixture loading ──────────────────────────────────────────────────

fn load_fixtures(dir: &std::path::Path, max_bytes: usize) -> Vec<LoadedFixture> {
    let mut fixtures = Vec::new();
    let mut skipped = 0usize;

    for entry in WalkDir::new(dir)
        .min_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if !path
            .extension()
            .is_some_and(|ext| ext == "json")
        {
            continue;
        }
        let meta = match std::fs::metadata(path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let raw_size = meta.len() as usize;
        if max_bytes > 0 && raw_size > max_bytes {
            skipped += 1;
            continue;
        }
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                warn!("Could not read {}: {e}", path.display());
                continue;
            }
        };
        match serde_json::from_slice::<StatelessValidationFixture>(&bytes) {
            Ok(f) => {
                let n_state_nodes = f.stateless_input.witness.state.len();
                let n_codes = f.stateless_input.witness.codes.len();
                let n_txs = f.stateless_input.block.body.transactions.len();
                let category = classify_fixture(raw_size, n_state_nodes, n_txs);
                fixtures.push(LoadedFixture {
                    fixture: f,
                    raw_size,
                    n_state_nodes,
                    n_codes,
                    n_txs,
                    category,
                });
            }
            Err(e) => {
                warn!("Could not parse {}: {e}", path.display());
            }
        }
    }

    if skipped > 0 {
        info!("Skipped {skipped} fixtures exceeding {max_bytes} bytes");
    }
    info!(
        "Loaded {} fixtures ({:.1} MB total)",
        fixtures.len(),
        fixtures.iter().map(|f| f.raw_size).sum::<usize>() as f64 / 1_048_576.0
    );
    fixtures
}

// ── Validation core ──────────────────────────────────────────────────

/// Run real Reth stateless validation. Returns (input_prep_ms, validation_ms).
fn run_validation(fixture: &StatelessValidationFixture) -> Result<(f64, f64)> {
    // Stage: create guest input (wraps StatelessInput for Reth guest)
    let t_prep = Instant::now();
    let input = StatelessValidatorRethInput::new(&fixture.stateless_input, fixture.success)
        .context("Failed to create Reth stateless validator input")?;
    let input_prep_ms = t_prep.elapsed().as_secs_f64() * 1000.0;

    // Stage: full validation — MPT proof verify, WitnessDB build,
    //        EVM execution, post-state root check
    let t_val = Instant::now();
    let _output = StatelessValidatorRethGuest::compute::<NoopPlatform>(input);
    let validation_ms = t_val.elapsed().as_secs_f64() * 1000.0;

    Ok((input_prep_ms, validation_ms))
}

// ── Handlers ─────────────────────────────────────────────────────────

/// `POST /witness` — accepts full fixture JSON, deserialises, validates.
async fn handle_witness(
    body: axum::body::Bytes,
) -> (StatusCode, axum::Json<ValidationResponse>) {
    let t_start = Instant::now();
    let body_len = body.len();

    let result = tokio::task::spawn_blocking(move || {
        // Deserialise
        let t0 = Instant::now();
        let fixture: StatelessValidationFixture = serde_json::from_slice(&body)?;
        let deser_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let name = fixture.name.clone();
        let gas = fixture.stateless_input.block.gas_used;

        let (input_prep_ms, validation_ms) = run_validation(&fixture)?;

        Ok::<_, anyhow::Error>((deser_ms, input_prep_ms, validation_ms, name, gas))
    })
    .await
    .unwrap();

    match result {
        Ok((deser_ms, input_prep_ms, validation_ms, name, gas)) => (
            StatusCode::OK,
            axum::Json(ValidationResponse {
                status: "accepted".into(),
                fixture_name: Some(name),
                witness_size_bytes: Some(body_len),
                block_gas_used: Some(gas),
                deser_ms: Some(deser_ms),
                input_prep_ms: Some(input_prep_ms),
                validation_ms: Some(validation_ms),
                total_server_ms: Some(t_start.elapsed().as_secs_f64() * 1000.0),
                ..Default::default()
            }),
        ),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            axum::Json(ValidationResponse {
                status: "rejected".into(),
                error: Some(format!("{e:#}")),
                witness_size_bytes: Some(body_len),
                total_server_ms: Some(t_start.elapsed().as_secs_f64() * 1000.0),
                ..Default::default()
            }),
        ),
    }
}

/// `POST /validate/{idx}` — validates pre-loaded fixture by index.
async fn handle_validate_by_idx(
    State(state): State<Arc<AppState>>,
    Path(idx): Path<usize>,
) -> (StatusCode, axum::Json<ValidationResponse>) {
    let t_start = Instant::now();

    if idx >= state.fixtures.len() {
        return (
            StatusCode::NOT_FOUND,
            axum::Json(ValidationResponse {
                status: "error".into(),
                error: Some(format!(
                    "fixture index {idx} out of range (have {})",
                    state.fixtures.len()
                )),
                ..Default::default()
            }),
        );
    }

    let loaded = &state.fixtures[idx];
    let name = loaded.fixture.name.clone();
    let gas = loaded.fixture.stateless_input.block.gas_used;
    let raw_size = loaded.raw_size;
    let n_state_nodes = loaded.n_state_nodes;
    let n_txs = loaded.n_txs;
    let category = loaded.category;

    // Clone the fixture for the blocking task
    let fixture = loaded.fixture.clone();
    let result = tokio::task::spawn_blocking(move || run_validation(&fixture)).await.unwrap();

    match result {
        Ok((input_prep_ms, validation_ms)) => (
            StatusCode::OK,
            axum::Json(ValidationResponse {
                status: "accepted".into(),
                fixture_name: Some(name),
                fixture_idx: Some(idx),
                witness_size_bytes: Some(raw_size),
                block_gas_used: Some(gas),
                n_state_nodes: Some(n_state_nodes),
                n_txs: Some(n_txs),
                category: Some(category.into()),
                deser_ms: Some(0.0), // pre-loaded, no deser cost
                input_prep_ms: Some(input_prep_ms),
                validation_ms: Some(validation_ms),
                total_server_ms: Some(t_start.elapsed().as_secs_f64() * 1000.0),
                ..Default::default()
            }),
        ),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            axum::Json(ValidationResponse {
                status: "rejected".into(),
                fixture_name: Some(name),
                fixture_idx: Some(idx),
                error: Some(format!("{e:#}")),
                total_server_ms: Some(t_start.elapsed().as_secs_f64() * 1000.0),
                ..Default::default()
            }),
        ),
    }
}

/// `GET /info` — fixture stats with per-category breakdown.
async fn handle_info(State(state): State<Arc<AppState>>) -> axum::Json<InfoResponse> {
    let total: usize = state.fixtures.iter().map(|f| f.raw_size).sum();
    let count = state.fixtures.len();

    let mut categories: std::collections::HashMap<String, (usize, usize, Vec<usize>)> =
        std::collections::HashMap::new();
    for (i, f) in state.fixtures.iter().enumerate() {
        let entry = categories
            .entry(f.category.to_string())
            .or_insert((0, 0, Vec::new()));
        entry.0 += 1;
        entry.1 += f.raw_size;
        entry.2.push(i);
    }
    let cat_stats = categories
        .into_iter()
        .map(|(k, (cnt, sz, indices))| {
            (
                k,
                CategoryStats {
                    count: cnt,
                    avg_size_kb: if cnt > 0 {
                        sz as f64 / cnt as f64 / 1024.0
                    } else {
                        0.0
                    },
                    indices,
                },
            )
        })
        .collect();

    axum::Json(InfoResponse {
        fixture_count: count,
        total_size_mb: total as f64 / 1_048_576.0,
        avg_size_kb: if count > 0 {
            total as f64 / count as f64 / 1024.0
        } else {
            0.0
        },
        categories: cat_stats,
    })
}

/// `GET /health` — liveness.
async fn handle_health() -> &'static str {
    "ok"
}

// ── main ─────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();

    let fixtures = load_fixtures(&args.fixtures_dir, args.max_fixture_bytes);
    let state = Arc::new(AppState { fixtures });

    let app = Router::new()
        .route("/witness", post(handle_witness))
        .route("/validate/{idx}", post(handle_validate_by_idx))
        .route("/info", get(handle_info))
        .route("/health", get(handle_health))
        .with_state(state);

    info!("BuilderTee server listening on {}", args.listen);
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

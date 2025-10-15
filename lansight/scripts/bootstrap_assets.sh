#!/usr/bin/env bash
# Bootstrap local assets (CLIP model, initial LLM weights for 512 & 768, datasets JSONL + images)
#
# Typical usage:
#   bash lansight/scripts/bootstrap_assets.sh
#
# Options (env vars):
#   CLIP_SOURCE=ms                           # prefer ModelScope mirror for CLIP
#   LLM_SIZES="512,768"                      # which LLM sizes to download (comma-separated)
#   SKIP_CLIP=1 / SKIP_LLM=1 / SKIP_DATA=1   # skip parts if already present
#   LLM_URL_512=...  LLM_URL_768=...         # override default LLM URLs per size
#   FORCE_HTTP1=1                            # force curl to use HTTP/1.1
#   SPEED_LIMIT=8192 SPEED_TIME=120          # low-speed abort tuning

set -euo pipefail

detect_root() {
  local here; here="$(cd "$(dirname "$0")" && pwd)"
  local candidates=(
    "$here/../.."
    "$PWD"
    "$PWD/.."
    "$PWD/../.."
  )
  for cand in "${candidates[@]}"; do
    if [ -d "$cand/lansight/model" ]; then
      echo "$(cd "$cand" && pwd)"; return 0
    fi
  done
  # fallback to script-based two-level up
  echo "$(cd "$(dirname "$0")/../.." && pwd)"
}

ROOT_DIR="${ROOT_DIR:-$(detect_root)}"
CLIP_DIR="$ROOT_DIR/lansight/model/vision_model/clip-vit-base-patch16"
DATA_DIR="$ROOT_DIR/datasets"
OUT_DIR="$ROOT_DIR/out"

LLM_SIZES_CSV="${LLM_SIZES:-512,768}"
IFS=',' read -r -a LLM_SIZES_ARR <<< "$LLM_SIZES_CSV"
CLIP_SOURCE="${CLIP_SOURCE:-hf}" # hf|ms
SKIP_CLIP="${SKIP_CLIP:-0}"
SKIP_LLM="${SKIP_LLM:-0}"
SKIP_DATA="${SKIP_DATA:-0}"

SFT_JSONL_URL="${SFT_JSONL_URL:-}"
PRETRAIN_JSONL_URL="${PRETRAIN_JSONL_URL:-}"
SFT_ZIP_URL="${SFT_ZIP_URL:-}"
PRETRAIN_ZIP_URL="${PRETRAIN_ZIP_URL:-}"

_has() { command -v "$1" >/dev/null 2>&1; }

# resume-friendly fetch (aria2c > curl > wget)
SPEED_LIMIT="${SPEED_LIMIT:-16384}"   # bytes/sec; abort if below this for SPEED_TIME seconds
SPEED_TIME="${SPEED_TIME:-60}"        # seconds of low speed before abort

_fetch_once() {
  # _fetch_once <url> <dest>
  local url="$1"; shift
  local dest="$1"; shift
  mkdir -p "$(dirname "$dest")"
  # prefer aria2c for robust multi-connection & resume
  if _has aria2c; then
    aria2c -c -x 8 -s 8 -k 1M -o "$(basename "$dest")" -d "$(dirname "$dest")" "$url"
  elif _has curl; then
    # resume (-C -), follow redirects, retry all errors, fail on HTTP errors, low-speed abort
    curl -L --retry 5 --retry-all-errors --connect-timeout 30 \
         --speed-time "$SPEED_TIME" --speed-limit "$SPEED_LIMIT" \
         ${FORCE_HTTP1:+--http1.1} -C - --fail -o "$dest" "$url"
  elif _has wget; then
    # continue (-c), more tries, generous timeouts
    wget -c --tries=5 --timeout=300 --read-timeout=300 -O "$dest" "$url"
  else
    echo "[ERROR] need aria2c or curl or wget to download: $url" >&2
    return 1
  fi
}

_try_fetch() {
  # _try_fetch "url1 url2 ..." dest
  local urls=( $1 ); shift || true
  local dest="$1"; shift || true
  local ok=1
  for u in "${urls[@]}"; do
    echo "--> fetch: $u"
    if _fetch_once "$u" "$dest"; then ok=0; break; fi
    echo "   retry next source..."
  done
  return $ok
}

_note() { echo -e "\033[33m[NOTE]\033[0m $*"; }
_ok()   { echo -e "\033[32m[OK]\033[0m   $*"; }
_err()  { echo -e "\033[31m[ERR]\033[0m  $*"; }

echo "==> Prepare folders under: $ROOT_DIR"
mkdir -p "$CLIP_DIR" "$DATA_DIR" "$OUT_DIR"

echo "==> Download CLIP (source=$CLIP_SOURCE)"
if [ "$SKIP_CLIP" = "1" ]; then
  _note "skip CLIP by SKIP_CLIP=1"
elif [ -f "$CLIP_DIR/pytorch_model.bin" ]; then
  _ok "CLIP already present: $CLIP_DIR"
else
  hf_base="https://huggingface.co/openai/clip-vit-base-patch16/resolve/main"
  ms_base="https://www.modelscope.cn/models/openai-mirror/clip-vit-base-patch16/resolve/master"
  # prefer source, fallback to the other
  if [ "$CLIP_SOURCE" = "hf" ]; then
    bases=("$hf_base" "$ms_base")
  else
    bases=("$ms_base" "$hf_base")
  fi
  files=(
    config.json
    preprocessor_config.json
    pytorch_model.bin
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    vocab.json
    merges.txt
  )
  for f in "${files[@]}"; do
    urls=("${bases[0]}/$f" "${bases[1]}/$f")
    if ! _try_fetch "${urls[*]}" "$CLIP_DIR/$f"; then
      _err "Failed to download CLIP file: $f"; exit 1
    fi
  done
  _ok "CLIP downloaded to $CLIP_DIR"
fi

echo "==> Download initial LLM(s): ${LLM_SIZES_ARR[*]}"
if [ "$SKIP_LLM" = "1" ]; then
  _note "skip LLM by SKIP_LLM=1"
else
  for sz in "${LLM_SIZES_ARR[@]}"; do
    dest="$OUT_DIR/llm_${sz}.pth"
    if [ -f "$dest" ]; then _ok "LLM present: $dest"; continue; fi
    urls=()
    # per-size override
    eval override="\${LLM_URL_${sz}:-}"
    if [ -n "$override" ]; then urls+=("$override"); fi
    urls+=(
      "https://huggingface.co/Timber0914/Lansight/resolve/main/llm_${sz}.pth"
      "https://hf-mirror.com/Timber0914/Lansight/resolve/main/llm_${sz}.pth"
    )
    if ! _try_fetch "${urls[*]}" "$dest"; then _err "Failed LLM ${sz}"; exit 1; fi
    _ok "LLM saved: $dest"
  done
fi

echo "==> Datasets"

dl_jsonl() {
  # dl_jsonl <name> <dest> <urls...>
  local name="$1"; shift
  local dest="$1"; shift
  local urls=( "$@" )
  if [ -f "$dest" ]; then _ok "$name already exists: $dest"; return 0; fi
  if [ ${#urls[@]} -eq 0 ]; then _note "skip $name (no URL provided)"; return 0; fi
  if _try_fetch "${urls[*]}" "$dest"; then _ok "$name downloaded: $dest"; else _err "fail $name"; fi
}

ensure_flat_layout() {
  # ensure_flat_layout <extracted_dir> <target_dir> <expected_subfolder>
  local extracted="$1"; shift
  local target="$1"; shift
  local expected="$1"; shift
  mkdir -p "$target"
  local src="$extracted"
  if [ -d "$extracted/$expected" ]; then src="$extracted/$expected"; fi
  # Use rsync or tar pipe to avoid ARG_MAX / too many files issues
  if _has rsync; then
    rsync -a "$src"/ "$target"/
  else
    tar -C "$src" -cf - . | tar -C "$target" -xf -
  fi
}

dl_zip_flat() {
  # dl_zip_flat <name> <zip_dest> <tmp_extract_dir> <target_dir> <expected_subfolder> <urls...>
  local name="$1"; shift
  local zipdest="$1"; shift
  local tmpdir="$1"; shift
  local target="$1"; shift
  local expected="$1"; shift
  local urls=( "$@" )

  if [ -d "$target" ] && [ "$(ls -A "$target" 2>/dev/null | wc -l)" -gt 0 ]; then _ok "$name already exists: $target"; return 0; fi
  if [ ${#urls[@]} -eq 0 ]; then _note "skip $name (no URL provided)"; return 0; fi
  if ! _try_fetch "${urls[*]}" "$zipdest"; then _err "fail $name"; return 1; fi
  if ! _has unzip; then _err "unzip not found, cannot extract $zipdest"; return 1; fi
  rm -rf "$tmpdir"
  mkdir -p "$tmpdir" "$target"
  unzip -q -o "$zipdest" -d "$tmpdir"
  ensure_flat_layout "$tmpdir" "$target" "$expected"
  # verify non-empty target; keep tmp/zip if empty for manual recovery
  if [ "$(find "$target" -type f 2>/dev/null | wc -l)" -gt 0 ]; then
    rm -rf "$tmpdir" "$zipdest"
    _ok "$name ready: $target"
  else
    _err "$name appears empty at $target; leaving temp and zip for manual check"
    return 1
  fi
}

# Default dataset URLs (can be overridden by env)
SFT_JSONL_URL_DEFAULT="https://huggingface.co/datasets/Timber0914/Lansight_datasets/resolve/main/sft_data.jsonl"
PRETRAIN_JSONL_URL_DEFAULT="https://huggingface.co/datasets/Timber0914/Lansight_datasets/resolve/main/pretrain_data.jsonl"
SFT_ZIP_URL_DEFAULT="https://huggingface.co/datasets/Timber0914/Lansight_datasets/resolve/main/sft_images.zip"
PRETRAIN_ZIP_URL_DEFAULT="https://huggingface.co/datasets/Timber0914/Lansight_datasets/resolve/main/pretrain_images.zip"

# Mirror fallbacks
SFT_JSONL_URLS=( "${SFT_JSONL_URL:-}" "$SFT_JSONL_URL_DEFAULT" "https://hf-mirror.com/datasets/Timber0914/Lansight_datasets/resolve/main/sft_data.jsonl" )
PRETRAIN_JSONL_URLS=( "${PRETRAIN_JSONL_URL:-}" "$PRETRAIN_JSONL_URL_DEFAULT" "https://hf-mirror.com/datasets/Timber0914/Lansight_datasets/resolve/main/pretrain_data.jsonl" )
SFT_ZIP_URLS=( "${SFT_ZIP_URL:-}" "$SFT_ZIP_URL_DEFAULT" "https://hf-mirror.com/datasets/Timber0914/Lansight_datasets/resolve/main/sft_images.zip" )
PRETRAIN_ZIP_URLS( ) { echo "${PRETRAIN_ZIP_URL:-}"; echo "$PRETRAIN_ZIP_URL_DEFAULT"; echo "https://hf-mirror.com/datasets/Timber0914/Lansight_datasets/resolve/main/pretrain_images.zip"; }

if [ "$SKIP_DATA" = "1" ]; then
  _note "skip datasets by SKIP_DATA=1"
else
  dl_jsonl "sft_data.jsonl" "$DATA_DIR/sft_data.jsonl" "${SFT_JSONL_URLS[@]}"
  dl_jsonl "pretrain_data.jsonl" "$DATA_DIR/pretrain_data.jsonl" "${PRETRAIN_JSONL_URLS[@]}"
  # Images (flat folders): datasets/sft_images and datasets/pretrain_images
  dl_zip_flat "SFT images" "$DATA_DIR/sft_images.zip" "$DATA_DIR/.tmp_sft_images" "$DATA_DIR/sft_images" "sft_images" "${SFT_ZIP_URLS[@]}"
  dl_zip_flat "Pretrain images" "$DATA_DIR/pretrain_images.zip" "$DATA_DIR/.tmp_pretrain_images" "$DATA_DIR/pretrain_images" "pretrain_images" "$(PRETRAIN_ZIP_URLS)"
fi

echo "==> Summary"
du -sh "$CLIP_DIR" 2>/dev/null || true
[ -f "${LLM_DEST:-}" ] && ls -lh "$LLM_DEST" || true
[ -f "$DATA_DIR/sft_data.jsonl" ] && ls -lh "$DATA_DIR/sft_data.jsonl" || true
[ -f "$DATA_DIR/pretrain_data.jsonl" ] && ls -lh "$DATA_DIR/pretrain_data.jsonl" || true
[ -d "$DATA_DIR/sft_images" ] && echo "sft_images -> $DATA_DIR/sft_images (files: $(find "$DATA_DIR/sft_images" -type f 2>/dev/null | wc -l))" || true
[ -d "$DATA_DIR/pretrain_images" ] && echo "pretrain_images -> $DATA_DIR/pretrain_images (files: $(find "$DATA_DIR/pretrain_images" -type f 2>/dev/null | wc -l))" || true

_ok "Bootstrap finished. Next: run training with python -m lansight.trainer.train_pretrain / train_sft"

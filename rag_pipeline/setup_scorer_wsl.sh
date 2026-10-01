#!/bin/bash
# Scorer environment inside WSL (Ubuntu), no sudo. Used for the Spark-based languages (Spanish, Amharic),
# because Spark on native Windows needs Hadoop's winutils. Same package versions as the Windows scorer env.
# Installs into ~/blend_scorer: user-space Java 11, a Python venv.
# pyspark/spark-nlp/py4j are installed from local files (PyPI's file server is unreliable on our network):
#   C:\Users\Siddharth\blend_scorer\wheels\  (spark_nlp, py4j)  +  pyspark tarball in Downloads.
set -euo pipefail

ROOT="$HOME/blend_scorer"
ENV="$ROOT/venv"
WHEELS="/mnt/c/Users/Siddharth/blend_scorer/wheels"
PYSPARK_TGZ="${PYSPARK_TGZ:-/mnt/c/Users/Siddharth/Downloads/pyspark-3.5.3 (1).tar.gz}"
mkdir -p "$ROOT"

# 1. Java 11 (Eclipse Temurin), user-space
if [ ! -x "$ROOT/jdk11/bin/java" ]; then
  echo "[1/4] Java 11"
  curl -sSL --retry 5 "https://api.adoptium.net/v3/binary/latest/11/ga/linux/x64/jdk/hotspot/normal/eclipse" -o "$ROOT/jdk11.tar.gz"
  mkdir -p "$ROOT/jdk11" && tar xzf "$ROOT/jdk11.tar.gz" -C "$ROOT/jdk11" --strip-components=1 && rm "$ROOT/jdk11.tar.gz"
fi
"$ROOT/jdk11/bin/java" -version 2>&1 | head -1

# 2. Python venv (Ubuntu's python3 has no ensurepip -> bootstrap pip)
if [ ! -x "$ENV/bin/pip" ]; then
  echo "[2/4] venv"
  python3 -m venv --without-pip "$ENV"
  curl -sS --retry 5 https://bootstrap.pypa.io/get-pip.py | "$ENV/bin/python" - -q
fi

# 3. Spark from local files, then the rest (same versions as the Windows scorer env)
echo "[3/4] packages"
"$ENV/bin/pip" install -q --retries 20 --timeout 120 --find-links "$WHEELS" "$PYSPARK_TGZ" "spark-nlp==5.5.3"
"$ENV/bin/pip" install -q --retries 20 --timeout 120 pandas==2.2.3 tqdm spacy==3.7.4 konlpy
"$ENV/bin/pip" install -q --retries 20 --timeout 120 \
  "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"

# 4. Smoke test: start Spark NLP (first run downloads its Java libraries from Maven)
echo "[4/4] Spark NLP start test"
JAVA_HOME="$ROOT/jdk11" "$ENV/bin/python" -c "import sparknlp; s = sparknlp.start(); print('Spark', s.version, '| Spark NLP', sparknlp.version())" 2>&1 | tail -1
echo "DONE: $ROOT"

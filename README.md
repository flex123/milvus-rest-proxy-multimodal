# Milvus 2.6.7 Native Hybrid Search REST Proxy

A production-grade REST API for Milvus 2.6.7 supporting native hybrid search with text and image embeddings.

## Features

- **Native Hybrid Search** - Uses Milvus 2.6's built-in `hybrid_search()` method
- **Multi-modal Search** - Combine text and image search with intelligent weighting
- **Dynamic Collections** - Search any collection via URL parameter
- **Flexible Filtering** - Support for price, bedroom, property type filters
- **CLIP Integration** - Text-to-image search using CLIP model
- **RRF & Weighted Ranking** - Choose between reciprocal rank fusion or custom weights

## Configuration

### Required Environment Variable

The proxy requires `MILVUS_HOST` to be set. Without it, the application will exit with an error.

```bash
# Option 1: Environment variable (for local development)
export MILVUS_HOST=your-milvus-host
python3 rest-proxy-multimodal.py

# Option 2: Inline (for local development)
MILVUS_HOST=your-milvus-host python3 rest-proxy-multimodal.py

# Option 3: Railway deployment (set in Variables tab)
# MILVUS_HOST=your-milvus-host
```

### Optional Environment Variables

```bash
MILVUS_PORT=443  # Default: 443
```

### Examples for Different Milvus Instances

```bash
# Local Milvus
MILVUS_HOST=localhost MILVUS_PORT=19530 python3 rest-proxy-multimodal.py

# Milvus Cloud
MILVUS_HOST=in01-xxxxxxxx.aws-us-west-2.vectordb.zilliz.com python3 rest-proxy-multimodal.py

# Railway Deployment
# Set MILVUS_HOST in Railway Variables tab, not in start command
```

## API Endpoints

### Hybrid Search
```bash
POST /search/{collection_name}/hybrid
Content-Type: application/json

{
    "text": "modern house with pool",
    "limit": 10,
    "filters": {
        "price_max": 500000,
        "bedrooms_min": 3
    }
}
```

### Text-only Search
```bash
POST /search/{collection_name}
Content-Type: application/json

{
    "text": "kitchen renovation",
    "limit": 5
}
```

### Collection Stats
```bash
GET /stats/{collection_name}
```

### List Collections
```bash
GET /collections
```

### Health Check
```bash
GET /health
```

## Railway Deployment

### Prerequisites
- GitHub repository with the proxy code
- Railway account

### Deployment Steps

1. **Create New Railway Project**:
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your `milvus-rest-proxy-multimodal` repository

2. **Configure Environment Variables** (Required):

   **Method A: Using Project Shared Variables (Recommended)**
   - Go to Project Settings → Variables tab
   - Add: `MILVUS_HOST=your-actual-milvus-host`
   - Optional: `MILVUS_PORT=443` (default)
   - Go to your service → Settings → Variables tab
   - Click "Import from project" and select the shared variables

   **Method B: Service-Specific Variables**
   - Go to your service → Settings → Variables tab
   - Add: `MILVUS_HOST=your-actual-milvus-host`
   - Optional: `MILVUS_PORT=443` (default)

   **Examples**:
   - Local: `MILVUS_HOST=localhost`
   - Milvus Cloud: `MILVUS_HOST=in01-xxxxxxxx.aws-us-west-2.vectordb.zilliz.com`

3. **Deployment Configuration**:
   - Railway automatically detects the configuration from `railway.toml`
   - Health check path: `/health`
   - Port: 8080
   - Uses NIXPACKS builder

4. **Monitoring**:
   - Check deployment logs in Railway dashboard
   - Health checks run every 5 minutes
   - Service restarts on failure (up to 10 retries)

### Troubleshooting
If deployment fails:
1. **Check Environment Variables**: Ensure `MILVUS_HOST` is properly set
   - Recommended: Use project shared variables and import them to your service
   - Verify the variable appears in your service's Variables tab
2. Verify the Milvus instance is accessible from Railway
3. Check deployment logs for detailed error messages
4. **Common Error**: "MILVUS_HOST environment variable is required" means the variable is not being passed to the container

### Alternative: Custom Start Command
Instead of using Variables, you can set the start command in Railway:
```
MILVUS_HOST=your-milvus-host bash start.sh
```

## How It Works

1. **Intelligent Weight Distribution**:
   - Text queries: 70% text, 30% image
   - Visual queries ("red roof"): 50/50 split
   - Custom weights available via API

2. **Collection Support**:
   - Dynamic collection names via URL
   - Works with any collection that has `text_embedding` and `image_embedding` fields
   - Supports all Milvus 2.6.7 collection types

3. **Filter Conversion**:
   - Converts JSON filters to Milvus expressions
   - Supports range queries (`_max`, `_min` suffixes)
   - Example: `{"price_max": 500000}` → `price <= 500000`

## Dependencies

- Python 3.8+
- Milvus 2.6.7
- Flask
- sentence-transformers
- torch
- transformers
- pymilvus

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
MILVUS_HOST=localhost python3 rest-proxy-multimodal.py
```

## Production Considerations

- **Model Loading**: Models load lazily on first request
- **Performance**: Sub-second response times for typical queries
- **Scalability**: Handles millions of vectors with Milvus clustering
- **Security**: Uses secure connections (TLS) by default
#!/usr/bin/env python3
"""
REST Proxy for Milvus 2.6.7 Native Hybrid Search
Implements hybrid search using AnnSearchRequest and RRFRanker
"""

from flask import Flask, request, jsonify
from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker, WeightedRanker
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
MILVUS_HOST = os.getenv('MILVUS_HOST')
if not MILVUS_HOST:
    raise ValueError(
        "MILVUS_HOST environment variable is required.\n"
        "Set it via:\n"
        "  - Environment variable: export MILVUS_HOST=your-milvus-host\n"
        "  - Railway Variables tab in dashboard\n"
        "  - Railway Start Command: MILVUS_HOST=your-host python3 rest-proxy-multimodal.py"
    )
MILVUS_PORT = int(os.getenv('MILVUS_PORT', 443))
TEXT_DIM = 512
IMAGE_DIM = 512

# Models
text_model = None
clip_model = None

def connect_to_milvus():
    """Connect to Milvus"""
    try:
        connections.connect(
            alias='default',
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            secure=True,
            timeout=30
        )
        logger.info(f"✅ Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to connect to Milvus: {e}")
        return False

def load_models():
    """Load embedding models"""
    global text_model, clip_model

    try:
        logger.info("📦 Loading embedding models...")
        text_model = SentenceTransformer('all-MiniLM-L6-v2')
        clip_model = SentenceTransformer('clip-ViT-B-32')
        logger.info("✅ Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        return False

def generate_text_embedding(text: str) -> np.ndarray:
    """Generate text embedding"""
    if text_model is None:
        if not load_models():
            raise Exception("Failed to load text model")

    # Use sentence-transformer for text
    embedding = text_model.encode(text)
    # Project to 512-dim to match image embedding
    if embedding.shape[0] > 512:
        embedding = embedding[:512]
    elif embedding.shape[0] < 512:
        embedding = np.pad(embedding, (0, 512 - embedding.shape[0]))

    return embedding.astype(np.float32)

def generate_image_embedding(image_url: str) -> Optional[np.ndarray]:
    """Generate image embedding using CLIP"""
    if clip_model is None:
        if not load_models():
            raise Exception("Failed to load CLIP model")

    try:
        import requests
        from PIL import Image
        from io import BytesIO

        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Use CLIP for image embedding (512-dim)
        embedding = clip_model.encode(image)
        return embedding.astype(np.float32)
    except Exception as e:
        logger.warning(f"Failed to encode image {image_url}: {e}")
        # Return zero vector
        return np.zeros(512, dtype=np.float32)

def generate_text_to_image_embedding(text: str) -> np.ndarray:
    """Generate image embedding from text using CLIP's text encoder"""
    if clip_model is None:
        if not load_models():
            raise Exception("Failed to load CLIP model")

    # Use CLIP's text encoder to simulate image embedding
    embedding = clip_model.encode([text])[0]
    return embedding.astype(np.float32)

def choose_ranker(query_text: str, data: Dict[str, Any]):
    """
    Intelligently choose the ranker based on query type
    Visual queries get more weight on image embeddings
    """
    # Visual-specific keywords that should emphasize image similarity
    visual_keywords = [
        'red', 'blue', 'green', 'yellow', 'white', 'black',  # Colors
        'modern', 'contemporary', 'traditional', 'rustic',  # Styles
        'kitchen', 'bathroom', 'fireplace', 'pool', 'garden',  # Features
        'roof', 'window', 'door', 'floor', 'wall', 'ceiling',  # Parts
        'view', 'ocean', 'mountain', 'city', 'lake',  # Views
        'spacious', 'cozy', 'luxury', 'minimalist',  # Descriptors
        'brick', 'wood', 'stone', 'glass', 'metal'  # Materials
    ]

    # Check if query contains visual keywords
    query_lower = query_text.lower()
    has_visual_keywords = any(keyword in query_lower for keyword in visual_keywords)

    # Check if there's an image URL (for image-based queries)
    has_image_url = data.get('image_url') and len(data.get('image_url', '')) > 0

    # Choose weights based on query type
    if has_image_url:
        # Image-based query: emphasize image similarity
        return WeightedRanker(0.2, 0.8)  # 20% text, 80% image
    elif has_visual_keywords:
        # Visual text query: high emphasis on image
        return WeightedRanker(0.2, 0.8)  # 20% text, 80% image
    else:
        # General query: balanced approach
        return RRFRanker()

def convert_filters_to_milvus_expr(filters: Dict[str, Any]) -> str:
    """Convert JSON filters to Milvus expression format"""
    if not filters:
        return None

    expressions = []

    for field, condition in filters.items():
        # Handle simple suffix format (price_max, price_min, etc.)
        if field.endswith('_max'):
            base_field = field[:-4]  # Remove '_max' suffix
            expressions.append(f'{base_field} <= {condition}')
        elif field.endswith('_min'):
            base_field = field[:-4]  # Remove '_min' suffix
            expressions.append(f'{base_field} >= {condition}')
        elif isinstance(condition, dict):
            # Handle operators like $lt, $gt, $gte, $lte, $eq
            for op, value in condition.items():
                if op == "$lt":
                    expressions.append(f'{field} < {value}')
                elif op == "$lte":
                    expressions.append(f'{field} <= {value}')
                elif op == "$gt":
                    expressions.append(f'{field} > {value}')
                elif op == "$gte":
                    expressions.append(f'{field} >= {value}')
                elif op == "$eq":
                    if isinstance(value, str):
                        expressions.append(f'{field} == "{value}"')
                    else:
                        expressions.append(f'{field} == {value}')
                elif op == "$ne":
                    if isinstance(value, str):
                        expressions.append(f'{field} != "{value}"')
                    else:
                        expressions.append(f'{field} != {value}')
                elif op == "$in":
                    if isinstance(value, list):
                        str_values = ', '.join([f'"{v}"' for v in value if isinstance(v, str)])
                        num_values = ', '.join([str(v) for v in value if not isinstance(v, str)])
                        values = str_values if str_values else num_values
                        expressions.append(f'{field} in [{values}]')
        else:
            # Handle exact match (no operator)
            if isinstance(condition, str):
                expressions.append(f'{field} == "{condition}"')
            else:
                expressions.append(f'{field} == {condition}')

    return " and ".join(expressions) if expressions else None

def format_results(results: List, include_scores: bool = True) -> List[Dict[str, Any]]:
    """Format search results for JSON response"""
    formatted_results = []

    for result in results:
        formatted = {
            'id': result.id,
            'distance': float(result.distance) if include_scores else None
        }

        # Add entity fields
        if hasattr(result, 'entity') and result.entity:
            for key, value in result.entity.items():
                if key != 'text_embedding' and key != 'image_embedding':  # Skip embeddings
                    formatted[key] = value

        formatted_results.append(formatted)

    return formatted_results

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'search_type': 'native_hybrid'
    })

@app.route('/collections', methods=['GET'])
def list_collections():
    """List all collections"""
    try:
        from pymilvus import utility
        collections = utility.list_collections()
        return jsonify({
            'collections': collections,
            'current': COLLECTION_NAME
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search/<collection_name>/hybrid', methods=['POST'])
def hybrid_search(collection_name):
    """
    Native hybrid search endpoint using Milvus 2.6.7 hybrid_search()
    Supports both text and image queries with optional fusion

    Query types:
    - "hybrid": Combine text and image embeddings (default)
    - "text": Only text embedding search
    - "image": Only image embedding search
    """
    try:
        data = request.get_json()
        query_text = data.get('text', '')
        image_url = data.get('image_url', '')
        limit = data.get('limit', 10)
        # Support both 'filter' and 'filters' for metadata filtering
        raw_filters = data.get('filter', data.get('filters', None))
        # Convert JSON filters to Milvus expression format
        filter_expr = convert_filters_to_milvus_expr(raw_filters) if raw_filters else None
        query_type = data.get('query_type', 'hybrid').lower()  # Default to hybrid

        if not query_text and not image_url:
            return jsonify({'error': 'Either text or image_url must be provided'}), 400

        if query_type not in ['hybrid', 'text', 'image']:
            return jsonify({'error': 'query_type must be one of: hybrid, text, image'}), 400

        collection = Collection(collection_name)
        collection.load()

        search_requests = []

        # Text search request
        if query_text:
            text_embedding = generate_text_embedding(query_text)
            req_text = AnnSearchRequest(
                data=[text_embedding],
                anns_field="text_embedding",
                param={"metric_type": "IP", "params": {"ef": 64}},
                limit=int(limit * 2),  # Get more candidates for fusion
                expr=filter_expr  # Apply metadata filter
            )
            search_requests.append(req_text)
            logger.info(f"Added text search for: '{query_text}' with filter: {filter_expr}")

        # Image search request
        if image_url:
            # Generate image embedding
            image_embedding = generate_image_embedding(image_url)
            req_image = AnnSearchRequest(
                data=[image_embedding],
                anns_field="image_embedding",
                param={"metric_type": "IP", "params": {"ef": 64}},
                limit=int(limit * 2),  # Get more candidates for fusion
                expr=filter_expr  # Apply metadata filter
            )
            search_requests.append(req_image)
            logger.info(f"Added image search for: {image_url} with filter: {filter_expr}")

        # Text-to-image search (if no image_url provided but we want multi-modal)
        elif query_text and len(search_requests) == 1:
            image_embedding = generate_text_to_image_embedding(query_text)
            req_image = AnnSearchRequest(
                data=[image_embedding],
                anns_field="image_embedding",
                param={"metric_type": "IP", "params": {"ef": 64}},
                limit=int(limit * 2),
                expr=filter_expr  # Apply metadata filter
            )
            search_requests.append(req_image)
            logger.info(f"Added text-to-image search for: '{query_text}' with filter: {filter_expr}")

        if not search_requests:
            return jsonify({'error': 'No valid search requests created'}), 400

        # Initialize variables for response
        ranker_type = None
        ranker_weights = None
        rerank = None

        # Perform search based on query_type
        if query_type == 'text' or query_type == 'image':
            # Single modality search
            if query_type == 'text' and not query_text:
                return jsonify({'error': 'text query is required for text search'}), 400
            if query_type == 'image' and not image_url and not query_text:
                return jsonify({'error': 'Either image_url or text (for text-to-image) is required for image search'}), 400

            # Find the appropriate search request
            if query_type == 'text':
                req = next((r for r in search_requests if r.anns_field == 'text_embedding'), None)
            else:
                req = next((r for r in search_requests if r.anns_field == 'image_embedding'), None)
                if not req and query_text:  # Text-to-image search
                    image_embedding = generate_text_to_image_embedding(query_text)
                    req = AnnSearchRequest(
                        data=[image_embedding],
                        anns_field="image_embedding",
                        param={"metric_type": "IP", "params": {"ef": 64}},
                        limit=limit,
                        expr=filter_expr
                    )
                    search_requests = [req]

            if not req:
                return jsonify({'error': f'Could not create {query_type} search request'}), 400

            results = collection.search(
                data=req.data,
                anns_field=req.anns_field,
                param=req.param,
                limit=limit,
                output_fields=[
                    "title", "description", "price", "city", "state",
                    "bedrooms", "bathrooms", "square_feet", "property_type",
                    "has_pool", "has_garden", "has_ocean_view", "has_mountain_view",
                    "neighborhood", "address", "zip_code", "image_url", "year_built"
                ]
            )
            formatted_results = format_results(results[0])
            ranker_type = f"{query_type}_only"

        else:  # query_type == 'hybrid'
            # Hybrid search with intelligent fusion
            rerank = choose_ranker(query_text, data)

            # Log the chosen ranker type for debugging
            if isinstance(rerank, WeightedRanker):
                logger.info(f"Using WeightedRanker for visual query: '{query_text}'")

            results = collection.hybrid_search(
                search_requests,
                rerank=rerank,
                limit=limit
            )

            # Format results - hybrid search returns different structure
            formatted_results = []
            if results and len(results) > 0:
                for hit in results[0]:
                    formatted = {
                        'id': hit.id,
                        'distance': float(hit.distance)
                    }

                    # Get full entity data
                    entity_results = collection.query(
                        expr=f"id == {hit.id}",
                        output_fields=[
                            "title", "description", "price", "city", "state",
                            "bedrooms", "bathrooms", "square_feet", "property_type",
                            "has_pool", "has_garden", "has_ocean_view", "has_mountain_view",
                            "neighborhood", "address", "zip_code", "image_url", "year_built"
                        ]
                    )

                    if entity_results:
                        entity = entity_results[0]
                        for key, value in entity.items():
                            if key != 'id':
                                formatted[key] = value

                    formatted_results.append(formatted)

            # Determine which ranker was used
            ranker_type = "weighted" if isinstance(rerank, WeightedRanker) else "rrf"
            ranker_weights = [0.2, 0.8] if isinstance(rerank, WeightedRanker) else None

        response = {
            'results': formatted_results,
            'total': len(formatted_results),
            'collection': collection_name,
            'query': {
                'text': query_text,
                'image_url': image_url,
                'filter': filter_expr,
                'query_type': query_type
            },
            'search_type': 'native_hybrid',
            'ranker': ranker_type,
            'ranker_weights': ranker_weights
        }

        logger.info(f"✅ Hybrid search returned {len(formatted_results)} results")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/search/<collection_name>', methods=['POST'])
def search(collection_name):
    """
    Search endpoint for text-only search with optional metadata filtering
    """
    try:
        data = request.get_json()
        query_text = data.get('text', '')
        limit = data.get('limit', 10)
        # Support both 'filter' and 'filters' for metadata filtering
        raw_filters = data.get('filter', data.get('filters', None))
        # Convert JSON filters to Milvus expression format
        filter_expr = convert_filters_to_milvus_expr(raw_filters) if raw_filters else None

        if not query_text:
            return jsonify({'error': 'text query is required'}), 400

        collection = Collection(collection_name)
        collection.load()

        # Generate text embedding
        text_embedding = generate_text_embedding(query_text)

        # Search with optional filter
        results = collection.search(
            data=[text_embedding],
            anns_field="text_embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=limit,
            expr=filter_expr,  # Apply metadata filter
            output_fields=[
                "title", "description", "price", "city", "state",
                "bedrooms", "bathrooms", "square_feet", "property_type",
                "has_pool", "has_garden", "has_ocean_view", "has_mountain_view",
                "neighborhood", "address", "zip_code", "image_url", "year_built"
            ]
        )

        formatted_results = format_results(results[0])

        return jsonify({
            'results': formatted_results,
            'total': len(formatted_results),
            'collection': collection_name,
            'query': {
                'text': query_text,
                'filter': filter_expr
            },
            'search_type': 'text_only'
        })

    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats/<collection_name>', methods=['GET'])
def get_stats(collection_name):
    """Get collection statistics"""
    try:
        collection = Collection(collection_name)
        collection.load()

        stats = {
            'collection': collection_name,
            'entities': collection.num_entities,
            'fields': []
        }

        # Add field information
        for field in collection.schema.fields:
            field_info = {
                'name': field.name,
                'type': str(field.dtype)
            }
            if hasattr(field, 'dim'):
                field_info['dim'] = field.dim
            stats['fields'].append(field_info)

        # Add index information
        indexes = collection.indexes
        stats['indexes'] = [
            {
                'field': idx.field_name,
                'name': idx.index_name,
                'params': idx.params
            }
            for idx in indexes
        ]

        return jsonify(stats)

    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

# Backward compatibility endpoints that use default collection
@app.route('/search/hybrid', methods=['POST'])
def hybrid_search_default():
    """Hybrid search with default collection for backward compatibility"""
    return hybrid_search(COLLECTION_NAME)

@app.route('/search', methods=['POST'])
def search_default():
    """Search with default collection for backward compatibility"""
    return search(COLLECTION_NAME)

@app.route('/stats', methods=['GET'])
def get_stats_default():
    """Stats with default collection for backward compatibility"""
    return get_stats(COLLECTION_NAME)

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("="*70)
    logger.info("🚀 Starting Milvus 2.6.7 Native Hybrid Search REST Proxy")
    logger.info("📦 Dynamic collection support via URL")
    logger.info(f"🌐 Server: http://localhost:5555")
    logger.info("="*70)

    # Connect to Milvus
    if not connect_to_milvus():
        logger.error("❌ Failed to connect to Milvus. Exiting.")
        exit(1)

    # Models will be loaded lazily on first request
    logger.info("📦 Models will be loaded on first request")

    # Start Flask app
    port = int(os.environ.get('PORT', 5555))
    app.run(host='0.0.0.0', port=port, debug=False)
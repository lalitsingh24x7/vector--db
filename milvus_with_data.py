#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pymilvus import (
    connections, db,
    Collection, CollectionSchema, FieldSchema, DataType,
    utility
)


# In[2]:


from pymilvus import connections, db

connections.connect(
    alias="default",
    host="localhost",
    port=19530,
    db_name="rqm_db"
)

print("Databases:", db.list_database())
print("✓ Connected successfully")


# In[3]:


fields = [
    FieldSchema(name="id",           dtype=DataType.INT64,         is_primary=True, auto_id=True),
    FieldSchema(name="source_table", dtype=DataType.VARCHAR,       max_length=100),
    FieldSchema(name="extra_fields", dtype=DataType.VARCHAR,       max_length=2000),
    FieldSchema(name="description",  dtype=DataType.VARCHAR,       max_length=1000),
    FieldSchema(name="embedding",    dtype=DataType.FLOAT_VECTOR,  dim=384),
]


# In[4]:


schema = CollectionSchema(fields, description="Unified metadata collection — all source tables")
collection = Collection("rqm_metadata", schema)


# In[5]:


collection.create_index(
    "embedding",
    {
        "metric_type": "COSINE",
        "index_type":  "IVF_FLAT",
        "params":      {"nlist": 128},
    }
)


# In[6]:


print("Collections in rqm_db:", utility.list_collections())
print("✓ rqm_metadata created successfully")


# In[7]:


import pandas as pd


# In[9]:


instrument_df = pd.read_csv('C:\\Users\\LALIT\\workspace\\virtualenv\\instrument_metadata.csv', header=0)
mortgage_df = pd.read_csv("C:\\Users\\LALIT\\workspace\\virtualenv\\modgage_data.csv", header=0)
print(instrument_df.head(2))
print("---------")
print(mortgage_df.head(2))


# In[10]:


# Each table has its own template
templates = {
    'instrument_metadata': lambda r: (
        f"Ticker {r['ticker']} belongs to {r['benchmark_id']} benchmark. "
        f"Asset class: {r['asset_class']}, region: {r['region']}, "
        f"currency: {r['currency']}, active from {r['start_date']} to {r['end_date']}."
    ),
    'mortgage_data': lambda r: (
        f"Security {r['security_id']} with CUSIP {r['cusip']} is a {r['collateral_type']} "
        f"MBS issued by {r['issuer']}, coupon {r['coupon']}%, originated {r['origination_year']}, "
        f"maturing {r['maturity_date']}, region: {r['region']}, currency: {r['currency']}, "
        f"active from {r['start_date']} to {r['end_date']}."
    ),
}


# In[11]:


print(templates['instrument_metadata'](instrument_df.iloc[0].to_dict()))
print()
print(templates['mortgage_data'](mortgage_df.iloc[0].to_dict()))


# In[12]:


from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # outputs 384-dim vectors


# In[13]:


import json

# ── Descriptions ──────────────────────────────────────────────
instrument_descriptions = [
    templates['instrument_metadata'](row.to_dict())
    for _, row in instrument_df.iterrows()
]
mortgage_descriptions = [
    templates['mortgage_data'](row.to_dict())
    for _, row in mortgage_df.iterrows()
]

# ── Embeddings ────────────────────────────────────────────────
instrument_embeddings = embedder.encode(instrument_descriptions, show_progress_bar=True).tolist()
mortgage_embeddings   = embedder.encode(mortgage_descriptions,   show_progress_bar=True).tolist()

# ── Extra fields (full row as JSON) ───────────────────────────
instrument_extra_fields = [json.dumps(row.to_dict()) for _, row in instrument_df.iterrows()]
mortgage_extra_fields   = [json.dumps(row.to_dict()) for _, row in mortgage_df.iterrows()]

# ── Source table tags ─────────────────────────────────────────
instrument_source_tables = ['instrument_metadata'] * len(instrument_df)
mortgage_source_tables   = ['mortgage_data']       * len(mortgage_df)

# ── Sanity check ──────────────────────────────────────────────
print(f"instrument → {len(instrument_embeddings)} vectors of dim {len(instrument_embeddings[0])}")
print(f"mortgage   → {len(mortgage_embeddings)} vectors of dim {len(mortgage_embeddings[0])}")

print(f"\n--- instrument sample ---")
print(f"source_table : {instrument_source_tables[0]}")
print(f"extra_fields : {instrument_extra_fields[0]}")
print(f"description  : {instrument_descriptions[0]}")
print(f"embedding    : {instrument_embeddings[0][:5]}")

print(f"\n--- mortgage sample ---")
print(f"source_table : {mortgage_source_tables[0]}")
print(f"extra_fields : {mortgage_extra_fields[0]}")
print(f"description  : {mortgage_descriptions[0]}")
print(f"embedding    : {mortgage_embeddings[0][:5]}")


# In[14]:


collection = Collection('rqm_metadata')

# ── Insert instrument_metadata ────────────────────────────────
collection.insert([
    instrument_source_tables,
    instrument_extra_fields,
    instrument_descriptions,
    instrument_embeddings,
])
print(f"✓ Inserted {len(instrument_df)} instrument_metadata rows")

# ── Insert mortgage_data ──────────────────────────────────────
collection.insert([
    mortgage_source_tables,
    mortgage_extra_fields,
    mortgage_descriptions,
    mortgage_embeddings,
])
print(f"✓ Inserted {len(mortgage_df)} mortgage_data rows")

# ── Flush to persist ──────────────────────────────────────────
collection.flush()
print(f"\nTotal entities in rqm_metadata: {collection.num_entities}")


# In[15]:


# ── Load collection ───────────────────────────────────────────
collection = Collection('rqm_metadata')
collection.load()


# In[16]:


# ── Search function ───────────────────────────────────────────
def raw_search(question, top_k=3, source_table=None):
    query_vector = embedder.encode([question]).tolist()

    expr = f'source_table == "{source_table}"' if source_table else None

    results = collection.search(
        data=query_vector,
        anns_field='embedding',
        param={'metric_type': 'COSINE', 'params': {'nprobe': 10}},
        limit=top_k,
        expr=expr,
        output_fields=['source_table', 'extra_fields', 'description']
    )

    for hit in results[0]:
        print(f"score      : {round(hit.score, 4)}")
        print(f"source     : {hit.entity.get('source_table')}")
        print(f"description: {hit.entity.get('description')}")
        print("---")


# In[17]:


# print("=== Query 1: search across ALL tables ===")
# raw_search("Do we have ticker USLC0003?")

print("=== Query 1: search across ALL tables ===")
raw_search("Do we have ticker USLC0003?", top_k=1)


# In[18]:


# ── Hybrid Search: semantic + scalar filter ───────────────────
def hybrid_search(question, filters=None, top_k=3):
    """
    filters: dict of exact match conditions
    e.g. {"source_table": "mortgage_data"}
    e.g. {"source_table": "instrument_metadata"}
    """
    query_vector = embedder.encode([question]).tolist()

    # Build expr from filters dict
    if filters:
        expr = " and ".join([f'{k} == "{v}"' for k, v in filters.items()])
    else:
        expr = None

    results = collection.search(
        data=query_vector,
        anns_field='embedding',
        param={'metric_type': 'COSINE', 'params': {'nprobe': 10}},
        limit=top_k,
        expr=expr,
        output_fields=['source_table', 'extra_fields', 'description']
    )

    for hit in results[0]:
        print(f"score      : {round(hit.score, 4)}")
        print(f"source     : {hit.entity.get('source_table')}")
        print(f"description: {hit.entity.get('description')}")
        print("---")


# In[19]:


# # ── Test hybrid queries ───────────────────────────────────────
# print("=== Hybrid 1: semantic + filter by source_table ===")
# hybrid_search(
#     question="fixed rate securities",
#     filters={"source_table": "mortgage_data"}
# )

print("\n=== Hybrid 2: no filter — search all tables ===")
hybrid_search(
    question="US equity instruments active since 2000"
)


# In[20]:


from mcp_server.client import MCPClient
client = MCPClient()

# SQL
result = await client.execute_query("SELECT COUNT(*) FROM instrument_metadata WHERE region = 'US'")

# Group by
# result = await client.execute_query("SELECT asset_class, COUNT(*) as count FROM instrument_metadata GROUP BY asset_class")


# In[ ]:


result


# In[ ]:


# result = await client.execute_query("SELECT * FROM instrument_metadata WHERE region = 'US'  limit 19")


# In[9]:


# result


# In[11]:





# In[ ]:


# from langchain.agents import create_agent


# In[ ]:





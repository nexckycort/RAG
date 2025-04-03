import { readdir } from 'node:fs/promises';

import { PINECONE_INDEX_NAME } from './config/environment';
import { pc } from './config/pinecone';
import {
  type Embeddings,
  extractTextFromMarkdown,
  generateEmbeddings,
  splitTextIntoChunks,
} from './utils';

const indexName = PINECONE_INDEX_NAME;

async function ensureIndexExists() {
  const indexes = await pc.listIndexes();

  const indexExists = indexes.indexes?.some(
    (index) => index.name === indexName,
  );
  if (!indexExists) {
    await pc.createIndex({
      name: indexName,
      dimension: 384,
      metric: 'cosine',
      spec: {
        serverless: {
          cloud: 'aws',
          region: 'us-east-1',
        },
      },
    });
  }
}

async function storeInPinecone(chunks: string[], embeddings: Embeddings[]) {
  const index = pc.Index(indexName);

  const vectors = embeddings.map(({ id, embedding }, i) => ({
    id: id,
    values: embedding,
    metadata: { text: chunks[i] },
  }));

  await index.upsert(vectors);
}

async function processAndStoreDocument(filePath: string) {
  const text = await extractTextFromMarkdown(filePath);

  const chunks = splitTextIntoChunks(text);

  const embeddings = await generateEmbeddings(chunks);

  await storeInPinecone(chunks, embeddings);
}

async function main() {
  await ensureIndexExists();

  const files = await readdir('./uploads', { recursive: true });

  for await (const file of files) {
    await processAndStoreDocument(`./uploads/${file}`);
  }
}

await main();
console.log('Done');

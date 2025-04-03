import { readdir } from 'node:fs/promises';

import { PINECONE_INDEX_NAME } from './config/environment';
import { pc } from './config/pinecone';
import {
  type Embeddings,
  extractTextFromMarkdown,
  generateEmbedding,
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
    metadata: { text: chunks[i] ?? '' },
  }));

  await index.upsert(vectors);
}

async function processAndStoreDocument(filePath: string) {
  const text = await extractTextFromMarkdown(filePath);

  const chunks = splitTextIntoChunks(text);

  const embeddings = await generateEmbeddings(chunks);

  await storeInPinecone(chunks, embeddings);
}

async function ask(query: string) {
  const index = pc.Index(indexName);

  const embedding = await generateEmbedding(query);

  const searchResults = await index.query({
    vector: embedding,
    topK: 5,
    includeMetadata: true,
  });

  const retrievedTexts = searchResults.matches
    .map((match) => match.metadata?.text)
    .join('\n');

  const context = `
      Contexto: ${retrievedTexts}
      Pregunta: ${query}`;

  const prompt = `Eres un asistente de inteligencia artificial potente y con comportamiento humano. Tu objetivo es ayudar al usuario usando únicamente la información provista en el CONTEXT BLOCK.
      REGLAS:
      - No debes compartir enlaces o referencias que no estén explícitamente incluidas en el CONTEXT BLOCK.
      - No te disculpes por respuestas anteriores; si hay nueva información en el contexto, simplemente continúa.
      - Si el usuario menciona el "workspace" o "contexto", se refiere al contenido entre START CONTEXT BLOCK y END OF CONTEXT BLOCK.
      - Si encuentras una URL en el CONTEXT BLOCK, úsala como referencia en la respuesta con el siguiente formato: ([número de referencia](link)).
      - Si se solicita una cita textual, intenta proporcionar el enlace de la fuente original.
      - No inventes información. Solo responde si la información se encuentra directamente en el CONTEXT BLOCK.
      - No respondas preguntas que no estén relacionadas con el CONTEXT BLOCK.
      START CONTEXT BLOCK
      ${context}
      END OF CONTEXT BLOCK`;

  const response = await fetch('http://localhost:8000/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  });

  const data = await response.json();
  console.log('Respuesta del modelo:', data.response);
}

async function main() {
  if (false) {
    await ensureIndexExists();

    const files = await readdir('./uploads', { recursive: true });

    for await (const file of files) {
      await processAndStoreDocument(`./uploads/${file}`);
    }
  }

  await ask('¿Cuáles son los beneficios de una dieta basada en plantas?');
}

await main();
console.log('Done');

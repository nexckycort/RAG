import { randomUUIDv7 } from 'bun';
import { marked } from 'marked';
import { SERVER_URL } from './config/environment';

export function generateShortUUID() {
  const fullUUID = randomUUIDv7();
  return fullUUID.slice(-10);
}

export async function extractTextFromMarkdown(mdPath: string): Promise<string> {
  const mdText = await Bun.file(mdPath).text();
  const html = await marked.parse(mdText);
  return html.replace(/<[^>]*>/g, '');
}

export function splitTextIntoChunks(
  text: string,
  chunkSize = 500,
  overlap = 50,
): string[] {
  const chunks: string[] = [];
  let start = 0;

  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    chunks.push(text.slice(start, end));
    start += chunkSize - overlap;
  }

  return chunks;
}

export type Embeddings = {
  id: string;
  embedding: number[];
};

export async function generateEmbeddings(
  chunks: string[],
): Promise<Array<Embeddings>> {
  try {
    const response = await fetch(`${SERVER_URL}/embed-chunks`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ chunks }),
    });

    if (!response.ok) {
      throw new Error(
        `Error HTTP: ${response.status} - ${response.statusText}`,
      );
    }

    const data: { embeddings: number[][] } = await response.json();

    if (
      !Array.isArray(data.embeddings) ||
      !data.embeddings.every((embedding) => Array.isArray(embedding))
    ) {
      throw new Error('La respuesta no contiene un array de embeddings válido');
    }

    const result = data.embeddings.map((embedding) => ({
      id: generateShortUUID(),
      embedding,
    }));

    return result;
  } catch (error) {
    console.error('Error al generar los embeddings:', error.message);
    throw error;
  }
}

export async function generateEmbedding(
  text: string,
): Promise<Embeddings['embedding']> {
  try {
    const response = await fetch(`${SERVER_URL}/embed`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(
        `Error HTTP: ${response.status} - ${response.statusText}`,
      );
    }

    const data: { embedding: Embeddings['embedding'] } = await response.json();

    if (!Array.isArray(data.embedding)) {
      throw new Error('La respuesta no contiene un embedding válido');
    }

    return data.embedding;
  } catch (error) {
    console.error('Error al generar el embedding:', error.message);
    throw error;
  }
}

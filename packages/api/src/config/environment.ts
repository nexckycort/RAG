export const SERVER_URL = import.meta.env.SERVER_URL ?? '';
export const PINECONE_API_KEY = import.meta.env.PINECONE_API_KEY ?? '';
export let PINECONE_INDEX_NAME = import.meta.env.PINECONE_INDEX_NAME ?? '';

if (PINECONE_INDEX_NAME === '') {
  PINECONE_INDEX_NAME = 'quickstart';
  console.warn('PINECONE_INDEX_NAME environment variable not set');
}

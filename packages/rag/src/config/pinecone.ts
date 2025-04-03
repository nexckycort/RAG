import { Pinecone } from '@pinecone-database/pinecone';
import { PINECONE_API_KEY } from './environment';

export const pc = new Pinecone({
  apiKey: PINECONE_API_KEY,
});

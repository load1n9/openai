# unofficial Deno wrapper for the Open Ai api

### usage:
```ts
import { OpenAI } from 'https://deno.land/x/openai/mod.ts';

const instance = new OpenAI('YOUR_API_KEY');

console.log(await instance.createCompletion('The meaning of life is'))
```

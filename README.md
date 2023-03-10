# unofficial Deno wrapper for the Open Ai api

[![Tags](https://img.shields.io/github/release/load1n9/openai)](https://github.com/load1n9/openai/releases)
[![Doc](https://doc.deno.land/badge.svg)](https://doc.deno.land/https/deno.land/x/openai/mod.ts)
[![Checks](https://github.com/load1n9/openai/actions/workflows/ci.yml/badge.svg)](https://github.com/load1n9/openai/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/load1n9/openai)](https://github.com/load1n9/openai/blob/master/LICENSE)

### Usage

```ts
import { OpenAI } from "https://deno.land/x/openai/mod.ts";

const openAI = new OpenAI("YOUR_API_KEY");

const completion = await openAI.createCompletion({
  model: "davinci",
  prompt: "The meaning of life is",
});

console.log(completion.choices);
```

### Maintainers

- Dean Srebnik ([@load1n9](https://github.com/load1n9))
- Lino Le Van ([@lino-levan](https://github.com/lino-levan))

### License

MIT

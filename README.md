# Unofficial Deno wrapper for the Open AI API

[![Tags](https://img.shields.io/github/release/load1n9/openai)](https://github.com/load1n9/openai/releases)
[![Doc](https://doc.deno.land/badge.svg)](https://doc.deno.land/https/deno.land/x/openai/mod.ts)
[![Checks](https://github.com/load1n9/openai/actions/workflows/ci.yml/badge.svg)](https://github.com/load1n9/openai/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/load1n9/openai)](https://github.com/load1n9/openai/blob/master/LICENSE)

## Usage

Your Open AI Api key ([found here](https://beta.openai.com/account/api-keys)) is
needed for this library to work. We recommend setting it as an environment
variable. Here is a configuration example.

```ts
import { OpenAI } from "https://deno.land/x/openai/mod.ts";

const openAI = new OpenAI(Deno.env.get("YOUR_API_KEY")!);
```

### Completion

```ts
import { OpenAI } from "https://deno.land/x/openai/mod.ts";

const openAI = new OpenAI(Deno.env.get("YOUR_API_KEY")!);

const completion = await openAI.createCompletion({
  model: "davinci",
  prompt: "The meaning of life is",
});

console.log(completion.choices);
```

### Chat Completion

```ts
import { OpenAI } from "https://deno.land/x/openai/mod.ts";

const openAI = new OpenAI(Deno.env.get("YOUR_API_KEY")!);

const chatCompletion = await openAI.createChatCompletion({
  model: "gpt-3.5-turbo",
  messages: [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Who won the world series in 2020?" },
    {
      "role": "assistant",
      "content": "The Los Angeles Dodgers won the World Series in 2020.",
    },
    { "role": "user", "content": "Where was it played?" },
  ],
});

console.log(chatCompletion);
```

### Image

```ts
import { OpenAI } from "https://deno.land/x/openai/mod.ts";

const openAI = new OpenAI(Deno.env.get("YOUR_API_KEY")!);

const image = await openAI.createImage({
  prompt: "A unicorn in space",
});

console.log(image);
```

### Edit

```ts
import { OpenAI } from "https://deno.land/x/openai/mod.ts";

const openAI = new OpenAI(Deno.env.get("YOUR_API_KEY")!);

const edit = await openAI.createEdit({
  model: "text-davinci-edit-001",
  input: "What day of the wek is it?",
  instruction: "Fix the spelling mistakes",
});

console.log(edit);
```

### Image Edit

```ts
import { OpenAI } from "https://deno.land/x/openai/mod.ts";

const openAI = new OpenAI(Deno.env.get("YOUR_API_KEY")!);

const imageEdit = await openAI.createImageEdit({
  image: "@otter.png",
  mask: "@mask.png",
  prompt: "A cute baby sea otter wearing a beret",
  n: 2,
  size: "1024x1024",
});

console.log(imageEdit);
```

### Image Variation

```ts
import { OpenAI } from "https://deno.land/x/openai/mod.ts";

const openAI = new OpenAI(Deno.env.get("YOUR_API_KEY")!);

const imageVariation = await openAI.createImageVariation({
  image: "@otter.png",
  n: 2,
  size: "1024x1024",
});

console.log(imageVariation);
```

## Maintainers

- Dean Srebnik ([@load1n9](https://github.com/load1n9))
- Lino Le Van ([@lino-levan](https://github.com/lino-levan))

## License

MIT

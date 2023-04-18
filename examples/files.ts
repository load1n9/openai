import { OpenAI } from "../mod.ts";

const openAI = new OpenAI(
  "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",
);

// TODO: Do this more portably
console.log(await openAI.uploadFile("./testdata/example.jsonl", "fine-tune"));

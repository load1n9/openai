import { OpenAI } from "../mod.ts";

const openAI = new OpenAI(Deno.env.get("YOUR_API_KEY")!);

// TODO: Do this more portably
console.log(await openAI.uploadFile("./testdata/example.jsonl", "fine-tune"));

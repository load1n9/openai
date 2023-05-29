import { OpenAI } from "../mod.ts";

const openAI = new OpenAI(Deno.env.get("YOUR_API_KEY")!);

openAI.createCompletionStream({
  model: "davinci",
  prompt: "The meaning of life is",
}, (chunk) => {
  console.log(chunk);
});

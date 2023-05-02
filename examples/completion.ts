import { OpenAI } from "../mod.ts";

const openAI = new OpenAI(Deno.env.get("YOUR_API_KEY")!);

const completion = await openAI.createCompletion({
  model: "davinci",
  prompt: "The meaning of life is",
});

console.log(completion);

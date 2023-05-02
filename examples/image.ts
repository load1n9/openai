import { OpenAI } from "../mod.ts";

const openAI = new OpenAI(Deno.env.get("YOUR_API_KEY")!);

const image = await openAI.createImage({
  prompt: "A unicorn in space",
});

console.log(image);

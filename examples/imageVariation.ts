import { OpenAI } from "../mod.ts";

const openAI = new OpenAI(Deno.env.get("YOUR_API_KEY")!);

const imageVariation = await openAI.createImageVariation({
  image: "@otter.png",
  n: 2,
  size: "1024x1024",
});

console.log(imageVariation);

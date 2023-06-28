import { OpenAI } from "../mod.ts";

const openAI = new OpenAI(Deno.env.get("YOUR_API_KEY")!);

const imageBuffer = await Deno.open("@otter.png", { read: true });
const imageVariation = await openAI.createImageVariation({
  image: imageBuffer,
  n: 2,
  size: "1024x1024",
});

console.log(imageVariation);

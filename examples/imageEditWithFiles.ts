import { OpenAI } from "../mod.ts";

const openAI = new OpenAI(Deno.env.get("YOUR_API_KEY")!);

const imageBuffer = await Deno.open("@otter.png", { read: true });
const maskBuffer = await Deno.open("@mask.png", { read: true });
const imageEdit = await openAI.createImageEdit({
  image: imageBuffer,
  mask: maskBuffer,
  prompt: "A cute baby sea otter wearing a beret",
  n: 2,
  size: "1024x1024",
});

console.log(imageEdit);

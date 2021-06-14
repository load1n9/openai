import { OpenAI } from '../mod.ts';

const instance = new OpenAI('YOUR_API_KEY');


console.log(await instance.createClassification(
    [
        ["A happy moment", "Positive"],
        ["I am sad.", "Negative"],
        ["I am feeling awesome", "Positive"]
    ],
    "It is a raining day :(",
    ["Positive", "Negative", "Neutral"]
))
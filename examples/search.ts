import { OpenAI } from '../mod.ts';

const instance = new OpenAI('YOUR_API_KEY');

console.log(await instance.createSearch(['White house', 'hospital', 'school'],'the president'))

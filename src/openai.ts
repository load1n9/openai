export class OpenAI {
  private privKey: string;

  constructor(privateKey: string) {
    this.privKey = privateKey;
  }

  public async createCompletion(
    prompt: string,
    engine = "davinci",
    temperature = 0.3,
    maxTokens = 60,
    topP = 1,
    frequencyPenalty = 0.5,
    presencePenalty = 0,
  ): Promise<Response> {
    const response = await fetch(
      `https://api.openai.com/v1/engines/${engine}/completions`,
      {
        body: JSON.stringify({
          prompt: prompt,
          temperature: temperature,
          max_tokens: maxTokens,
          top_p: topP,
          frequency_penalty: frequencyPenalty,
          presence_penalty: presencePenalty,
        }),
        headers: {
          Authorization: `Bearer ${this.privKey}`,
          "Content-Type": "application/json",
        },
        method: "POST",
      },
    );
    return response.json();
  }

  public async createSearch(
    documents: string[],
    query: string,
    engine = "davinci",
  ): Promise<Response> {
    const response = await fetch(
      `https://api.openai.com/v1/engines/${engine}/search`,
      {
        body: JSON.stringify({
          documents: documents,
          query: query,
        }),
        headers: {
          Authorization: `Bearer ${this.privKey}`,
          "Content-Type": "application/json",
        },
        method: "POST",
      },
    );
    return response.json();
  }

  public async createClassification(
    examples: string[][],
    query: string,
    labels: string[],
    model = "curie",
    searchModel = "ada",
  ): Promise<Response> {
    const response = await fetch("https://api.openai.com/v1/classifications", {
      body: JSON.stringify({
        examples: examples,
        query: query,
        search_model: searchModel,
        model: model,
        labels: labels,
      }),
      headers: {
        Authorization: `Bearer ${this.privKey}`,
        "Content-Type": "application/json",
      },
      method: "POST",
    });
    return response.json();
  }

  public async createAnswer(
    documents: string[],
    question: string,
    examplesContext: string,
    examples: string[][],
    maxTokens = 5,
    model = "curie",
    searchModel = "ada",
  ): Promise<Response> {
    const response = await fetch("https://api.openai.com/v1/classifications", {
      body: JSON.stringify({
        documents: documents,
        question: question,
        search_model: searchModel,
        model: model,
        examples_context: examplesContext,
        examples: examples,
        max_tokens: maxTokens,
      }),
      headers: {
        Authorization: `Bearer ${this.privKey}`,
        "Content-Type": "application/json",
      },
      method: "POST",
    });
    return response.json();
  }

  public async retrieveEngine(engine: string): Promise<Response> {
    const response = await fetch(
      `https://api.openai.com/v1/engines/${engine}`,
      {
        headers: {
          Authorization: `Bearer ${this.privKey}`,
        },
      },
    );
    return response.json();
  }

  public async listEngines(): Promise<Response> {
    const response = await fetch("https://api.openai.com/v1/engines", {
      headers: {
        Authorization: `Bearer ${this.privKey}`,
      },
    });
    return response.json();
  }
  public async listFiles(): Promise<Response> {
    const response = await fetch("https://api.openai.com/v1/files", {
      headers: {
        Authorization: `Bearer ${this.privKey}`,
      },
    });
    return response.json();
  }
}

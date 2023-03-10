import type {
  ChatCompletion,
  ChatCompletionOptions,
  Completion,
  CompletionOptions,
  DeletedFile,
  DeletedFineTune,
  Edit,
  EditOptions,
  Embedding,
  EmbeddingsOptions,
  File,
  FileList,
  FineTune,
  FineTuneEvent,
  FineTuneEventList,
  FineTuneList,
  FineTuneOptions,
  Image,
  ImageEditOptions,
  ImageOptions,
  ImageVariationOptions,
  Model,
  ModelList,
  Moderation,
  Transcription,
  TranscriptionOptions,
  Translation,
  TranslationOptions,
} from "./types.ts";

const baseUrl = "https://api.openai.com/v1";

export class OpenAI {
  #privateKey: string;

  constructor(privateKey: string) {
    this.#privateKey = privateKey;
  }

  // deno-lint-ignore no-explicit-any
  async #request(url: string, body: any, method = "POST") {
    const response = await fetch(`${baseUrl}${url}`, {
      body: body ? JSON.stringify(body) : undefined,
      headers: {
        Authorization: `Bearer ${this.#privateKey}`,
        "Content-Type": "application/json",
      },
      method,
    });

    return await response.json();
  }

  /**
   * Lists the currently available models, and provides basic information about each one such as the owner and availability.
   *
   * https://platform.openai.com/docs/api-reference/models/list
   */
  async listModels(): Promise<ModelList> {
    return await this.#request("/models", undefined, "GET");
  }

  /**
   * Retrieves a model instance, providing basic information about the model such as the owner and permissioning.
   *
   * https://platform.openai.com/docs/api-reference/models/retrieve
   */
  async getModel(model: string): Promise<Model> {
    return await this.#request(`/models/${model}`, undefined, "GET");
  }

  /**
   * Creates a completion for the provided prompt and parameters
   *
   * https://platform.openai.com/docs/api-reference/completions/create
   */
  async createCompletion(options: CompletionOptions): Promise<Completion> {
    // TODO: make options.stream work
    return await this.#request(`/completions`, {
      model: options.model,
      prompt: options.prompt,
      suffix: options.suffix,
      max_tokens: options.maxTokens,
      temperature: options.temperature,
      top_p: options.topP,
      n: options.n,
      stream: options.stream,
      logprobs: options.logprobs,
      echo: options.echo,
      stop: options.stop,
      presence_penalty: options.presencePenalty,
      frequency_penalty: options.frequencyPenalty,
      best_of: options.bestOf,
      logit_bias: options.logitBias,
      user: options.user,
    });
  }

  /**
   * Creates a completion for the chat message
   *
   * https://platform.openai.com/docs/api-reference/chat/create
   */
  async createChatCompletion(
    options: ChatCompletionOptions,
  ): Promise<ChatCompletion> {
    return await this.#request(`/chat/completions`, {
      model: options.model,
      messages: options.messages,
      temperature: options.temperature,
      top_p: options.topP,
      n: options.n,
      stream: options.stream,
      stop: options.stop,
      max_tokens: options.maxTokens,
      presence_penalty: options.presencePenalty,
      frequency_penalty: options.frequencyPenalty,
      logit_bias: options.logitBias,
      user: options.user,
    });
  }

  /**
   * Creates a new edit for the provided input, instruction, and parameters.
   *
   * https://platform.openai.com/docs/api-reference/edits/create
   */
  async createEdit(options: EditOptions): Promise<Edit> {
    return await this.#request(`/edits`, {
      model: options.model,
      input: options.input,
      instruction: options.instruction,
      n: options.n,
      temperature: options.temperature,
      top_p: options.topP,
    });
  }

  /**
   * Creates an image given a prompt.
   *
   * https://platform.openai.com/docs/api-reference/images/create
   */
  async createImage(options: ImageOptions): Promise<Image> {
    return await this.#request(`/images/generations`, {
      prompt: options.prompt,
      n: options.n,
      size: options.size,
      response_format: options.responseFormat,
      user: options.user,
    });
  }

  /**
   * Creates an edited or extended image given an original image and a prompt.
   *
   * https://platform.openai.com/docs/api-reference/images/create-edit
   */
  async createImageEdit(options: ImageEditOptions): Promise<Image> {
    return await this.#request(`/images/edits`, {
      image: options.image,
      mask: options.mask,
      prompt: options.prompt,
      n: options.n,
      size: options.size,
      response_format: options.responseFormat,
      user: options.user,
    });
  }

  /**
   * Creates a variation of a given image.
   *
   * https://platform.openai.com/docs/api-reference/images/create-variation
   */
  async createImageVariation(options: ImageVariationOptions): Promise<Image> {
    return await this.#request(`/images/variations`, {
      image: options.image,
      n: options.n,
      size: options.size,
      response_format: options.responseFormat,
      user: options.user,
    });
  }

  /**
   * Creates an embedding vector representing the input text.
   *
   * https://platform.openai.com/docs/api-reference/embeddings/create
   */
  async createEmbeddings(options: EmbeddingsOptions): Promise<Embedding> {
    return await this.#request(`/embeddings`, options);
  }

  /**
   * Transcribes audio into the input language.
   *
   * https://platform.openai.com/docs/api-reference/audio/create
   */
  async createTranscription(
    options: TranscriptionOptions,
  ): Promise<Transcription> {
    return await this.#request(`/audio/transcriptions`, {
      file: options.file,
      model: options.model,
      prompt: options.prompt,
      response_format: options.responseFormat,
      temperature: options.temperature,
      language: options.language,
    });
  }

  /**
   * Translates audio into into English.
   *
   * https://platform.openai.com/docs/api-reference/audio/create
   */
  async createTranslation(options: TranslationOptions): Promise<Translation> {
    return await this.#request(`/audio/translations`, {
      file: options.file,
      model: options.model,
      prompt: options.prompt,
      response_format: options.responseFormat,
      temperature: options.temperature,
    });
  }

  /**
   * Returns a list of files that belong to the user's organization.
   *
   * https://platform.openai.com/docs/api-reference/files/list
   */
  async listFiles(): Promise<FileList> {
    return await this.#request(`/files`, undefined, "GET");
  }

  /**
   * Upload a file that contains document(s) to be used across various endpoints/features. Currently, the size of all the files uploaded by one organization can be up to 1 GB. Please contact us if you need to increase the storage limit.
   *
   * https://platform.openai.com/docs/api-reference/files/upload
   */
  async uploadFile(file: string, purpose: string): Promise<File> {
    return await this.#request(`/files`, {
      file,
      purpose,
    });
  }

  /**
   * Delete a file.
   *
   * https://platform.openai.com/docs/api-reference/files/delete
   */
  async deleteFile(fileId: string): Promise<DeletedFile> {
    return await this.#request(`/files/${fileId}`, undefined, "DELETE");
  }

  /**
   * Returns information about a specific file.
   *
   * https://platform.openai.com/docs/api-reference/files/retrieve
   */
  async retrieveFile(fileId: string): Promise<File> {
    return await this.#request(`/files/${fileId}`, undefined, "GET");
  }

  /**
   * Returns the contents of the specified file
   *
   * https://platform.openai.com/docs/api-reference/files/retrieve-content
   */
  async retrieveFileContent(fileId: string) {
    const response = await fetch(`${baseUrl}/files/${fileId}/content`, {
      headers: {
        Authorization: `Bearer ${this.#privateKey}`,
        "Content-Type": "application/json",
      },
    });
    return response.body;
  }

  /**
   * Creates a job that fine-tunes a specified model from a given dataset.
   *
   * https://platform.openai.com/docs/api-reference/fine-tunes/create
   */
  async createFineTune(
    options: FineTuneOptions,
  ): Promise<(FineTune & { events: FineTuneEvent[] })> {
    return await this.#request(`/fine-tunes`, {
      training_file: options.trainingFile,
      validation_file: options.validationFile,
      model: options.model,
      n_epochs: options.nEpochs,
      batch_size: options.batchSize,
      learning_rate_multiplier: options.learningRateMultiplier,
      prompt_loss_weight: options.promptLossWeight,
      compute_classification_metrics: options.computeClassificationMetrics,
      classification_n_classes: options.classificationNClasses,
      classification_positive_class: options.classificationPositiveClass,
      classification_betas: options.classificationBetas,
      suffix: options.suffix,
    });
  }

  /**
   * List your organization's fine-tuning jobs
   *
   * https://platform.openai.com/docs/api-reference/fine-tunes/list
   */
  async listFineTunes(): Promise<FineTuneList> {
    return await this.#request(`/fine-tunes`, undefined, "GET");
  }

  /**
   * Gets info about the fine-tune job.
   *
   * https://platform.openai.com/docs/api-reference/fine-tunes/retrieve
   */
  async retrieveFineTune(
    fineTuneId: string,
  ): Promise<(FineTune & { events: FineTuneEvent[] })> {
    return await this.#request(`/fine-tunes/${fineTuneId}`, undefined, "GET");
  }

  /**
   * Immediately cancel a fine-tune job.
   *
   * https://platform.openai.com/docs/api-reference/fine-tunes/cancel
   */
  async cancelFineTune(
    fineTuneId: string,
  ): Promise<(FineTune & { events: FineTuneEvent[] })> {
    return await this.#request(`/fine-tunes/${fineTuneId}/cancel`, undefined);
  }

  /**
   * Get fine-grained status updates for a fine-tune job.
   *
   * https://platform.openai.com/docs/api-reference/fine-tunes/events
   */
  async listFineTuneEvents(fineTuneId: string): Promise<FineTuneEventList> {
    // TODO: stream query parameter
    return await this.#request(
      `/fine-tunes/${fineTuneId}/events`,
      undefined,
      "GET",
    );
  }

  /**
   * Delete a fine-tuned model. You must have the Owner role in your organization.
   *
   * https://platform.openai.com/docs/api-reference/fine-tunes/delete-model
   */
  async deleteFineTuneModel(model: string): Promise<DeletedFineTune> {
    return await this.#request(`/models/${model}`, undefined, "DELETE");
  }

  /**
   * Classifies if text violates OpenAI's Content Policy
   *
   * https://platform.openai.com/docs/api-reference/moderations/create
   */
  async createModeration(
    input: string | string[],
    model?: string,
  ): Promise<Moderation> {
    return await this.#request(`/moderations`, {
      input,
      model,
    });
  }
}

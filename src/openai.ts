import { basename } from "https://deno.land/std@0.189.0/path/mod.ts";
import { decodeStream, throwError } from "./util.ts";
import type {
  ChatCompletion,
  ChatCompletionOptions,
  ChatCompletionStream,
  Completion,
  CompletionOptions,
  CompletionStream,
  DeletedFile,
  DeletedFineTune,
  Edit,
  EditOptions,
  Embedding,
  EmbeddingsOptions,
  FileInstance,
  FileList,
  FileSpecifier,
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

const defaultBaseUrl = "https://api.openai.com/v1";

export class OpenAI {
  #privateKey: string;
  #baseUrl: string;

  constructor(privateKey: string, options?: { baseUrl?: string }) {
    this.#privateKey = privateKey;
    this.#baseUrl = options?.baseUrl ?? defaultBaseUrl;
  }

  async #request(
    url: string,
    // deno-lint-ignore no-explicit-any
    body: any,
    options?: { method?: string; noContentType?: boolean },
  ) {
    const response = await fetch(
      `${this.#baseUrl}${url}`,
      {
        body: options?.noContentType
          ? body
          : (body ? JSON.stringify(body) : undefined),
        headers: {
          Authorization: `Bearer ${this.#privateKey}`,
          ...(
            options?.noContentType ? {} : {
              "Content-Type": "application/json",
            }
          ),
        },
        method: options?.method ?? "POST",
      },
    );
    const data = await response.json();

    throwError(data);

    return data;
  }

  /**
   * Lists the currently available models, and provides basic information about each one such as the owner and availability.
   *
   * https://platform.openai.com/docs/api-reference/models/list
   */
  async listModels(): Promise<ModelList> {
    return await this.#request("/models", undefined, { method: "GET" });
  }

  /**
   * Retrieves a model instance, providing basic information about the model such as the owner and permissioning.
   *
   * https://platform.openai.com/docs/api-reference/models/retrieve
   */
  async getModel(model: string): Promise<Model> {
    return await this.#request(`/models/${model}`, undefined, {
      method: "GET",
    });
  }

  /**
   * Creates a completion for the provided prompt and parameters
   *
   * https://platform.openai.com/docs/api-reference/completions/create
   */
  async createCompletion(options: CompletionOptions): Promise<Completion> {
    return await this.#request(`/completions`, {
      model: options.model,
      prompt: options.prompt,
      suffix: options.suffix,
      max_tokens: options.maxTokens,
      temperature: options.temperature,
      top_p: options.topP,
      n: options.n,
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
   * Creates a completion stream for the provided prompt and parameters
   *
   * https://platform.openai.com/docs/api-reference/completions/create
   */
  async createCompletionStream(
    options: Omit<CompletionOptions, "bestOf">,
    callback: (chunk: CompletionStream) => void,
  ): Promise<void> {
    const res = await fetch(
      `${this.#baseUrl}/completions`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.#privateKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: options.model,
          prompt: options.prompt,
          suffix: options.suffix,
          max_tokens: options.maxTokens,
          temperature: options.temperature,
          top_p: options.topP,
          n: options.n,
          stream: true,
          logprobs: options.logprobs,
          echo: options.echo,
          stop: options.stop,
          presence_penalty: options.presencePenalty,
          frequency_penalty: options.frequencyPenalty,
          logit_bias: options.logitBias,
          user: options.user,
        }),
      },
    );

    await decodeStream(res, callback);
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
      stop: options.stop,
      max_tokens: options.maxTokens,
      presence_penalty: options.presencePenalty,
      frequency_penalty: options.frequencyPenalty,
      logit_bias: options.logitBias,
      user: options.user,
      functions: options.functions,
      function_call: options.function_call
    });
  }

  /**
   * Creates a completion stream for the chat message
   *
   * https://platform.openai.com/docs/api-reference/chat/create
   */
  async createChatCompletionStream(
    options: ChatCompletionOptions,
    callback: (chunk: ChatCompletionStream) => void,
  ): Promise<void> {
    const res = await fetch(
      `${this.#baseUrl}/chat/completions`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.#privateKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: options.model,
          messages: options.messages,
          temperature: options.temperature,
          top_p: options.topP,
          n: options.n,
          stream: true,
          stop: options.stop,
          max_tokens: options.maxTokens,
          presence_penalty: options.presencePenalty,
          frequency_penalty: options.frequencyPenalty,
          logit_bias: options.logitBias,
          user: options.user,
          functions: options.functions,
          function_call: options.function_call
        }),
      },
    );

    await decodeStream(res, callback);
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
    const formData = new FormData();

    // Model specified
    formData.append("model", options.model);

    // File data
    if (typeof options.file === "string") {
      const file = await Deno.readFile(options.file);

      formData.append(
        "file",
        new File([file], basename(options.file)),
      );
    } else {
      // Deno types are wrong
      formData.append("file", options.file as unknown as Blob);
    }

    if (options.prompt) {
      formData.append("prompt", options.prompt);
    }
    if (options.responseFormat) {
      formData.append("response_format", options.responseFormat);
    }
    if (options.temperature) {
      formData.append("temperature", options.temperature.toString());
    }
    if (options.language) {
      formData.append("language", options.language);
    }

    return await this.#request(`/audio/transcriptions`, formData, {
      noContentType: true,
    });
  }

  /**
   * Translates audio into into English.
   *
   * https://platform.openai.com/docs/api-reference/audio/create
   */
  async createTranslation(options: TranslationOptions): Promise<Translation> {
    const formData = new FormData();

    // Model specified
    formData.append("model", options.model);

    // File data
    if (typeof options.file === "string") {
      const file = await Deno.readFile(options.file);

      formData.append(
        "file",
        new File([file], basename(options.file)),
      );
    } else {
      // Deno types are wrong
      formData.append("file", options.file as unknown as Blob);
    }

    if (options.prompt) {
      formData.append("prompt", options.prompt);
    }
    if (options.responseFormat) {
      formData.append("response_format", options.responseFormat);
    }
    if (options.temperature) {
      formData.append("temperature", options.temperature.toString());
    }

    return await this.#request(`/audio/translations`, formData, {
      noContentType: true,
    });
  }

  /**
   * Returns a list of files that belong to the user's organization.
   *
   * https://platform.openai.com/docs/api-reference/files/list
   */
  async listFiles(): Promise<FileList> {
    return await this.#request(`/files`, undefined, { method: "GET" });
  }

  /**
   * Upload a file that contains document(s) to be used across various endpoints/features. Currently, the size of all the files uploaded by one organization can be up to 1 GB. Please contact us if you need to increase the storage limit.
   *
   * https://platform.openai.com/docs/api-reference/files/upload
   */
  async uploadFile(
    file: FileSpecifier,
    purpose: string,
  ): Promise<FileInstance> {
    const formData = new FormData();

    // Model specified
    formData.append("file", file);

    // File data
    if (typeof file === "string") {
      const fileData = await Deno.readFile(file);

      formData.append(
        "file",
        new File([fileData], basename(file)),
      );
    } else {
      // Deno types are wrong
      formData.append("file", file as unknown as Blob);
    }

    formData.append("purpose", purpose);

    return await this.#request(`/files`, formData, {
      noContentType: true,
      method: "POST",
    });
  }

  /**
   * Delete a file.
   *
   * https://platform.openai.com/docs/api-reference/files/delete
   */
  async deleteFile(fileId: string): Promise<DeletedFile> {
    return await this.#request(`/files/${fileId}`, undefined, {
      method: "DELETE",
    });
  }

  /**
   * Returns information about a specific file.
   *
   * https://platform.openai.com/docs/api-reference/files/retrieve
   */
  async retrieveFile(fileId: string): Promise<FileInstance> {
    return await this.#request(`/files/${fileId}`, undefined, {
      method: "GET",
    });
  }

  /**
   * Returns the contents of the specified file
   *
   * https://platform.openai.com/docs/api-reference/files/retrieve-content
   */
  async retrieveFileContent(fileId: string) {
    const response = await fetch(
      `${this.#baseUrl}/files/${fileId}/content`,
      {
        headers: {
          Authorization: `Bearer ${this.#privateKey}`,
          "Content-Type": "application/json",
        },
      },
    );
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
    return await this.#request(`/fine-tunes`, undefined, { method: "GET" });
  }

  /**
   * Gets info about the fine-tune job.
   *
   * https://platform.openai.com/docs/api-reference/fine-tunes/retrieve
   */
  async retrieveFineTune(
    fineTuneId: string,
  ): Promise<(FineTune & { events: FineTuneEvent[] })> {
    return await this.#request(`/fine-tunes/${fineTuneId}`, undefined, {
      method: "GET",
    });
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
      { method: "GET" },
    );
  }

  /**
   * Delete a fine-tuned model. You must have the Owner role in your organization.
   *
   * https://platform.openai.com/docs/api-reference/fine-tunes/delete-model
   */
  async deleteFineTuneModel(model: string): Promise<DeletedFineTune> {
    return await this.#request(`/models/${model}`, undefined, {
      method: "DELETE",
    });
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

# RAG-based Email Autoresponder

This repo contains a RAG-based email autoresponder system for the [European Masters in AI](https://muia.dia.fi.upm.es/en/) at UPM. A system overview, a response example, modifications to M3's InfoNCE-IBN loss, and a breakdown of the Modal applications for its running are shown below. A summary of the project from problem statement to system results is available in the [MSc AI defense slides](M_Sc_AI_Defense.pdf). The [full thesis](M_Sc_AI_Thesis.pdf) is also made available (to be replaced with UPM's OA link once published).

Note: GitHub does not render the slides/thesis very well; you can preview them here, but it's better to download them if you want an accurate version.

Note#2: This repo is archived. No further updates are expected.

## Overview

<img width="5097" height="4729" alt="muia_prod_pipeline" src="https://github.com/user-attachments/assets/e01e88dc-2b1e-4f5a-afdd-2b82038f7f48" />

## Anonymized examples (English/Spanish)

### Conditional admission follow-up

**Incoming email**

> I was just conditionally admitted to MUIA. Any update? I am worried about the date of my Bachelor degree being too late for MUIA and losing my spot. Can you please help me?
>
> On Thu, Jul 2, 2026 at 12:28 AM Ralph <[redacted email]> wrote:
>
> Dear Damiano,
>
> If I apply through Helios and am admitted to MUIA, but I do not complete my bachelor's degree until September, would I still be able to start the programme in September?
>
> Thank you for your help.
>
> Best regards,
>
> Ralph

<details>
<summary><strong>Internal reasoning trace</strong></summary>

```text
Damiano Zanardini, Director of the Master in Artificial Intelligence (MUIA) at UPM.
Ralph ([redacted email]).
The student (Ralph) has been conditionally admitted to MUIA. He is worried that if he completes his Bachelor's degree in September, it might be too late and he might lose his spot.

* Chunk 1: "Admitted with conditional enrollment" is normal for students finishing their degree. It is permitted to enroll even if the TFM and up to 9 ECTS are pending, provided the student finishes the degree before completing the Master's.
* Chunk 2: Conditional enrollment means the student accepts the condition that they must finish undergraduate studies before any postgraduate grades can be uploaded (specifically before the first useful call, e.g., January). If not completed, they lose enrollment/money.
* Chunk 3: Students who haven't completed their degree can reserve a spot conditionally. They must meet specific criteria by September 30 (e.g., official academic record/proof of credits).
* Chunk 4: Deadline for submitting additional documentation is 30 September 2026. Students who haven't completed their degree can enroll with conditional admission, provided they submit required documentation by 30 September 2026.
* Chunk 5: Mentions deadlines (some from 2023, so treat with caution as per instructions, but Chunk 4 specifically mentions 2026 for documentation).

* Ralph is admitted conditionally.
* He finishes his degree in September.
* The deadline for documentation is September 30.
* He can enroll conditionally as long as he finishes the degree before he finishes the Master's.
* He shouldn't lose his spot as long as he submits the necessary documents by the deadline (Sept 30).

* Reassure him that conditional admission is normal.
* Explain that completing the degree in September is acceptable.
* Emphasize the requirement to finish the Bachelor's before the Master's is completed.
* Remind him of the documentation deadline (September 30).

* Tone: Professional, concise, reassuring.
* Language: English (matches the original email).

* Draft:
    Dear Ralph,

    Please do not worry. Being admitted "with conditional enrollment" is a normal procedure for students who are still finishing their undergraduate studies.

    As long as you complete your Bachelor's degree and submit the required documentation by the September 30 deadline, you will be able to start the program. The main requirement is that you must finish your Bachelor's degree before you complete the Master's degree.

    Best regards,
    Damiano Zanardini

* Check against constraints: No subject line, no signature/greetings outside tags. Use <message> tags.
* Check language: English.
* Check content accuracy: Chunk 1 and 4 support that conditional admission is fine and that documentation is due by Sept 30.

Self-Correction during drafting: The user specifically asks if completing in September is "too late". I should clarify that as long as the degree is completed and documentation is submitted by the September 30 deadline, it is fine.

Refined Draft:

Dear Ralph,

Please do not worry. Being admitted with conditional enrollment is a normal procedure for students who are still in the process of finishing their degree.

As long as you complete your Bachelor's degree and submit the necessary documentation by the September 30 deadline, you will not lose your spot. The key requirement is that you must finish your undergraduate studies before you complete the Master's degree.

Best regards,
Damiano Zanardini

Wait, let's check Chunk 1 again regarding the "9 ECTS" and "TFM". It says it's permitted to enroll even with those pending. The core message is: finishing in September is fine, just hit the Sept 30 documentation deadline.

Final Polish:

Dear Ralph,

Please do not worry. Being admitted with conditional enrollment is a standard procedure for students who are still finishing their undergraduate degree.

As long as you complete your Bachelor's degree and submit the required documentation by the September 30 deadline, you will be able to start the program. The primary requirement is simply that you must complete your Bachelor's degree before you finish your Master's degree.

Best regards,
Damiano Zanardini
```

</details>

**Generated response**

> Dear Ralph,
>
> Please do not worry. Being admitted with conditional enrollment is a standard procedure for students who are still in the process of finishing their undergraduate studies.
>
> As long as you complete your Bachelor's degree and submit the required documentation by the September 30 deadline, you will be able to start the program. The essential requirement is that you must finish your Bachelor's degree before you complete your Master's degree.
>
> Best regards,
>
> Damiano Zanardini

### Schedule query

**Incoming email**

> Hola,
>
> ¿Cuándo se imparte Redes Bayesianas? No encuentro el horario en Moodle.
>
> Gracias.

**Generated response**

> Hola,
>
> El horario para la asignatura de Redes Bayesianas (A4) en el curso 2025-26 es el siguiente:
>
> - Martes: 09:00 – 10:00
> - Miércoles: 09:00 – 10:00
> - Jueves: 17:00 – 19:00
>
> Ten en cuenta que esta asignatura se imparte de forma concentrada durante las primeras ocho semanas del semestre.
>
> Saludos,
>
> Damiano Zanardini

## Fine-tuning

M3 is fine-tuned with its default InfoNCE-style loss and in-batch negatives (IBN):

```math
\mathcal{L}_{*,\mathrm{IBN}}
=
-\frac{1}{|\mathcal{B}_Q|}
\sum_{i=1}^{|\mathcal{B}_Q|}
\log
\frac{
\exp\left(s_*\left(q_i,p_i^{+}\right)/\tau\right)
}{
Z_i
}
```

To reduce the number of false negatives in softmax's denominator (due to topic collision and data augmentation), three increasigly more aggresive masking strategies are used.

### Exact positive-passage mask

Exact token-matching instances of the positive passage for query $q_i$ are masked from the in-batch positives and negatives:

```math
\begin{aligned}
Z_i ={}&
\exp\left(s_*\left(q_i,p_i^{+}\right)/\tau\right)
\\
&+
\sum_{p\in\mathcal{P}_i^{-}}
\exp\left(s_*\left(q_i,p\right)/\tau\right)
\\
&+
\sum_{\substack{j=1\\j\neq i}}^{|\mathcal{B}_Q|}
m^{\mathrm{pos}}_{i,p_j^{+}}
\exp\left(s_*\left(q_i,p_j^{+}\right)/\tau\right)
\\
&+
\sum_{\substack{j=1\\j\neq i}}^{|\mathcal{B}_Q|}
\sum_{p\in\mathcal{P}_j^{-}}
m^{\mathrm{pos}}_{i,p}
\exp\left(s_*\left(q_i,p\right)/\tau\right)
\end{aligned}
```

where the collision mask $m^{\mathrm{pos}}_{i,p}$ zeroes out any additional IBN term whose passage tokens exactly match those of the positive passage for $q_i$:

```math
m^{\mathrm{pos}}_{i,p}=
\begin{cases}
0 & \text{if } \mathrm{tok}(p)=\mathrm{tok}(p_i^{+}),\\
1 & \text{otherwise.}
\end{cases}
```

### Same-group positive-passage mask

Any passage matching a positive passage associated with the same query-expansion group as $q_i$ is masked:

```math
\begin{aligned}
Z_i ={}&
\exp\left(s_*\left(q_i,p_i^{+}\right)/\tau\right)
\\
&+
\sum_{p\in\mathcal{P}_i^{-}}
\exp\left(s_*\left(q_i,p\right)/\tau\right)
\\
&+
\sum_{\substack{j=1\\j\neq i}}^{|\mathcal{B}_Q|}
m^{\mathrm{group}}_{i,p_j^{+}}
\exp\left(s_*\left(q_i,p_j^{+}\right)/\tau\right)
\\
&+
\sum_{\substack{j=1\\j\neq i}}^{|\mathcal{B}_Q|}
\sum_{p\in\mathcal{P}_j^{-}}
m^{\mathrm{group}}_{i,p}
\exp\left(s_*\left(q_i,p\right)/\tau\right)
\end{aligned}
```

where the mask $m^{\mathrm{group}}_{i,p}$ zeroes out a passage whose tokens match a positive passage for any query in the same expansion group as $q_i$:

```math
m^{\mathrm{group}}_{i,p}=
\begin{cases}
0 & \text{if } \exists k\in\{1,\dots,|\mathcal{B}_Q|\}: g(q_k)=g(q_i) \text{ and } \mathrm{tok}(p)=\mathrm{tok}(p_k^{+}),\\
1 & \text{otherwise.}
\end{cases}
```

### Similar-group positive-passage mask

Additionally, positive passages associated with queries considered similar to $q_i$ by a cross-encoder are also masked:

```math
\begin{aligned}
Z_i ={}&
\exp\left(s_*\left(q_i,p_i^{+}\right)/\tau\right)
\\
&+
\sum_{p\in\mathcal{P}_i^{-}}
\exp\left(s_*\left(q_i,p\right)/\tau\right)
\\
&+
\sum_{\substack{j=1\\j\neq i}}^{|\mathcal{B}_Q|}
m^{\mathrm{sim\_group}}_{i,p_j^{+}}
\exp\left(s_*\left(q_i,p_j^{+}\right)/\tau\right)
\\
&+
\sum_{\substack{j=1\\j\neq i}}^{|\mathcal{B}_Q|}
\sum_{p\in\mathcal{P}_j^{-}}
m^{\mathrm{sim\_group}}_{i,p}
\exp\left(s_*\left(q_i,p\right)/\tau\right)
\end{aligned}
```

where the mask $m^{\mathrm{sim\_group}}_{i,p}$ zeroes out passages matching a positive from the same expansion group or from a query whose cross-encoder similarity score with $q_i$ is at least $0$:

```math
m^{\mathrm{sim\_group}}_{i,p}=
\begin{cases}
0 & \begin{aligned}
    &\text{if } \exists k\in\{1,\dots,|\mathcal{B}_Q|\}:\\
    &\left(g(q_k)=g(q_i) \text{ or } \mathrm{score}_{\mathrm{cross\text{-}encoder}}(q_i^r,q_k^r)\geq 0\right)\\
    &\text{and } \mathrm{tok}(p)=\mathrm{tok}(p_k^{+}),
\end{aligned}\\
1 & \text{otherwise.}
\end{cases}
```

## Modal Applications

The tables below summarize the remote Modal applications and functions used by the system.

### `crawler-agent`

| Function            | Schedule     | Hardware | Timeout (s) | Scaledown (s) | Volume |
| ------------------- | ------------ | -------- | ----------: | ------------: | ------ |
| `run_crawler_agent` | `0 9 10 9 *` | CPU      |       86400 |       Default | Yes    |

### `email-agent`

| Function          | Schedule                      | Hardware | Timeout (s) | Scaledown (s) | Volume |
| ----------------- | ----------------------------- | -------- | ----------: | ------------: | ------ |
| `run_email_agent` | `0 9 * * *` (`Europe/Madrid`) | CPU      |        5400 |       Default | No     |

### `decoder-legacy`

| Function                     | Schedule  | Hardware | Timeout (s) | Scaledown (s) | Volume |
| ---------------------------- | --------- | -------- | ----------: | ------------: | ------ |
| `run_local_lm_or_vlm_legacy` | On demand | L40S GPU |         900 |           180 | No     |

### `decoder-latest`

| Function                     | Schedule  | Hardware | Timeout (s) | Scaledown (s) | Volume |
| ---------------------------- | --------- | -------- | ----------: | ------------: | ------ |
| `run_local_lm_or_vlm_latest` | On demand | H100 GPU |        1800 |           180 | No     |

### `encoder-cpu`

| Function                                                    | Schedule  | Hardware | Timeout (s) | Scaledown (s) | Volume |
| ----------------------------------------------------------- | --------- | -------- | ----------: | ------------: | ------ |
| `run_encoder_cpu_batch_document_embedder`                   | On demand | CPU      |        3600 |            60 | Yes    |
| `run_encoder_cpu_batch_query_embedder_and_qdrant_retriever` | On demand | CPU      |        3600 |            60 | Yes    |

### `encoder-gpu`

| Function                                                    | Schedule  | Hardware | Timeout (s) | Scaledown (s) | Volume |
| ----------------------------------------------------------- | --------- | -------- | ----------: | ------------: | ------ |
| `run_encoder_gpu_batch_document_embedder`                   | On demand | L40S GPU |        1800 |            60 | Yes    |
| `run_encoder_gpu_batch_query_embedder_and_qdrant_retriever` | On demand | L40S GPU |        1800 |            60 | Yes    |
| `run_encoder_gpu_reranker`                                  | On demand | L40S GPU |        1800 |            60 | Yes    |

### `qdrant-server`

| Function              | Schedule     | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --------------------- | ------------ | -------- | ----------: | ------------: | ------ |
| `serve_qdrant_server` | Web endpoint | CPU      |        3600 |           900 | Yes    |

### `storage-handler`

| Function              | Schedule  | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --------------------- | --------- | -------- | ----------: | ------------: | ------ |
| `write_chunk_records` | On demand | CPU      |        3600 |            60 | Yes    |
| `read_jsonl_records`  | On demand | CPU      |        3600 |            60 | Yes    |

### `volume-handler`

| Function                 | Schedule  | Hardware | Timeout (s) | Scaledown (s) | Volume |
| ------------------------ | --------- | -------- | ----------: | ------------: | ------ |
| `delete_volume_folders`  | On demand | CPU      |        3600 |            60 | Yes    |
| `count_lm_output_tokens` | On demand | CPU      |        3600 |            60 | Yes    |

### `collection-handler`

| Function                          | Schedule  | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --------------------------------- | --------- | -------- | ----------: | ------------: | ------ |
| `drop_legacy_collections`         | On demand | CPU      |        3600 |            60 | No     |
| `create_collections`              | On demand | CPU      |        3600 |            60 | No     |
| `enable_collection_optimizations` | On demand | CPU      |        3600 |            60 | No     |
| `write_batch_points`              | On demand | CPU      |        3600 |            60 | No     |
| `dump_collection_payloads`        | On demand | CPU      |        3600 |            60 | No     |

### `decoder-latest-tokenizer`

| Function                        | Schedule  | Hardware | Timeout (s) | Scaledown (s) | Volume |
| ------------------------------- | --------- | -------- | ----------: | ------------: | ------ |
| `count_decoder_latest_tokens`   | On demand | CPU      |        1800 |            60 | No     |
| `truncate_decoder_latest_texts` | On demand | CPU      |        1800 |            60 | No     |

### `curator`

| Function                                    | Schedule  | Hardware | Timeout (s) | Scaledown (s) | Volume |
| ------------------------------------------- | --------- | -------- | ----------: | ------------: | ------ |
| `run_email_knowledge_base_curator_pipeline` | On demand | CPU      |       28800 |            60 | Yes    |

### `llm-judge`

| Function        | Schedule  | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --------------- | --------- | -------- | ----------: | ------------: | ------ |
| `run_llm_judge` | On demand | CPU      |         900 |            60 | No     |

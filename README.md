# RAG-based Email Autoresponder

This repo contains a RAG-based email autoresponder system for the [European Masters in AI](https://muia.dia.fi.upm.es/en/) at UPM. A system overview, response examples, modifications to M3's InfoNCE-IBN loss, and a breakdown of the Modal applications for its running are shown below. A summary of the project from problem statement to system results is available in the [MSc AI defense slides](M_Sc_AI_Defense.pdf). The [full thesis](M_Sc_AI_Thesis.pdf) is also made available (to be replaced with UPM's OA link once published).

Note: GitHub does not render the slides/thesis very well; you can preview them here, but it's better to download them if you want an accurate version.

## Overview

<img width="5097" height="4729" alt="muia_prod_pipeline" src="https://github.com/user-attachments/assets/e01e88dc-2b1e-4f5a-afdd-2b82038f7f48" />

## Anonymized examples

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

### Admission rejection

**Incoming email**

> Buenas tardes,
>
> Me gustaría saber cuáles son las razones por las que se ha denegado el acceso a este máster.
>
> Cordialmente
>
> Dario

**Generated response**

> Estimado Dario,
>
> Lamento informarte que el Máster Universitario en Inteligencia Artificial es un programa de alta demanda, con un número de solicitantes muy superior al de plazas disponibles.
>
> Las decisiones de admisión son tomadas por la Comisión Académica del Máster (CAMIA) siguiendo un baremo que prioriza perfiles con una formación sólida en informática y excelentes resultados académicos. Los motivos más frecuentes de denegación suelen ser:
>
> - El grado de origen no está alineado con la Ingeniería Informática (requisito estricto para mantener sellos de calidad como Euro-Inf).
> - La nota media académica no alcanza el nivel requerido.
> - La universidad de origen no cumple con los criterios de prestigio establecidos en nuestro baremo.
>
> Si consideras que tu perfil ha sido evaluado incorrectamente, te invitamos a responder a este correo aportando los argumentos necesarios para su revisión. Asimismo, si te interesa seguir formándote en nuestro departamento, te sugiero explorar el Máster en Ciencia de Datos (MUCD) o nuestros títulos profesionales en áreas de IA y Machine Learning.
>
> Atentamente,
>
> Damiano Zanardini
>
> Director del Máster Universitario en Inteligencia Artificial
>
> Universidad Politécnica de Madrid

### Degree certificate request

**Incoming email**

> Hola Damiano,
>
> ¿Me podrías decir cómo se solicita el título, tras haber defendido exitosamente el máster hace unos días?
>
> Gracias y buen verano,
>
> Verónica Real Malcruz

**Generated response**

> Hola Verónica,
>
> Enhorabuena por la defensa de tu TFM.
>
> Para solicitar el título, primero debes esperar a que el acta de la defensa sea recibida y procesada por la Secretaría de Alumnos para que tu expediente se cierre formalmente. Una vez hecho esto, puedes realizar la solicitud completando el formulario que encontrarás en el siguiente enlace: https://muia.dia.fi.upm.es/es/estructura/#solicitud_del_titulo
>
> Ten en cuenta que todo el proceso administrativo es gestionado directamente por la Secretaría (secretaria@fi.upm.es).
>
> Un saludo,
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

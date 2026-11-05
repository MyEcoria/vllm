# Ternary-Bonsai-2-27B (Q2_0) sur RX 9070 : ce qui a marché, ce qui n'a pas marché

Objectif de départ : **2000 t/s en prefill (pp) et 130 t/s en génération (tg)** pour
`prism-ml/Ternary-Bonsai-2-27B-gguf:Q2_0`, servi par vLLM, sur cette machine précise.

- **Prefill : atteint** (2048-2064 t/s sur 512 tokens).
- **Génération : atteinte en chat en moyenne** (jusqu'à 140,7 t/s), **pas sur tous les textes**
  (≈ 85-95 t/s sur de la prose longue, ≈ 105-113 t/s en complétion brute). Les raisons sont en §6.

Lancement : `bonsai launch` (voir §2). Point de départ : 367 t/s en prefill, 14,6 t/s en génération.

---

## 1. Matériel et logiciel

| | |
|---|---|
| GPU | AMD Radeon RX 9070 (gfx1201, RDNA4), 56 CU, 16 Go GDDR6, 640 Go/s théoriques, Infinity Cache 64 Mo |
| CPU / RAM | Ryzen 5 5600 (6 cœurs / 12 threads), 15 Go |
| OS / pilote | Ubuntu 26.10, ROCm système (HIP 7.15), `librocthrust-dev` installé (nécessaire pour compiler les noyaux HIP) |
| vLLM | 0.33.0rc1.dev455 (roue ROCm 10.0) dans `~/Desktop/Perso/vllm/.venv`, torch 2.13 ROCm, triton 3.8 |
| Plugin | `~/Desktop/Perso/vllm-gguf-plugin`, branche `bonsai-ternary-hadamard` (**non commitée**) |
| Modèle | Qwen3.8-27B (architecture `qwen35` : 48 couches Gated DeltaNet + 16 couches d'attention), poids ternaires {-1,0,1} × échelle fp16, pliés dans une base tournée (Hadamard 1024 + signes) |

Mesures de référence sur la carte :
- copie mémoire GPU : 564 Go/s (lecture + écriture) ; lecture seule atteinte par un noyau : ~630 Go/s ;
- GEMM fp16 (hipBLASLt / Triton) : ~105 TFLOPS ; int8 (`torch._int_mm` / Triton) : ~200-208 TOPS ;
- pendant la génération, la carte est à ses fréquences max (GPU ~2720-3095 MHz, mémoire 1258 MHz).

---

## 2. Utilisation

```bash
bonsai launch                 # au premier plan (Ctrl+C pour arrêter)
bonsai launch --detach        # en arrière-plan, attend que le serveur réponde, affiche les adresses
bonsai launch --status | --stop | --plan | --show-key
```

API compatible OpenAI sur le port 8000 (`/v1/chat/completions`, `/v1/completions`), modèle `bonsai`.
Le serveur écoute sur `0.0.0.0` avec une clé API (créée une fois, dans `~/.bonsai/api_key`,
`--no-api-key` pour s'en passer, `--host 127.0.0.1` pour rester en local). Le raisonnement du modèle
est séparé de la réponse (`reasoning_content`) par le parseur `qwen3`.

### Profils mesurés (via `bonsai`, 3 prompts de chat de 512 tokens générés)

| Commande | Contexte max par requête | pp512 | tg chat (moy.) | dissertation / explication / code | tg complétion brute |
|---|---|---|---|---|---|
| `bonsai launch` (défaut : KV fp8, 4 requêtes) | **63 488** | 2048 | **129,2** | 87 / 116 / 185 | 103,3 |
| `bonsai launch --kv bf16` | 30 720 | 2047 | **134,1** | 89 / 124 / 190 | 106,7 |
| `bonsai launch --max-seqs 1` | **69 632** | 2056 | 127,1 | 86 / 115 / 181 | 101,2 |
| meilleur réglage mesuré (bf16, contexte 4k, 4 requêtes) | 4 096 | 2056 | **140,7** | 95 / 127 / 200 | 113,4 |

### Contexte maximal en gardant tout dans la VRAM

Mesuré en demandant à vLLM la limite du modèle (262 144) : il répond la longueur maximale qui tient
avec `--gpu-memory-utilization 0.95`, poids, drafter et graphes compris :

| Cache KV | 1 requête à la fois | 4 requêtes à la fois |
|---|---|---|
| bf16 | 34 944 | 31 616 |
| **fp8** | **70 864** | 64 272 |

`bonsai` garde une petite marge : 69 632 / 63 488 en fp8, 34 816 / 30 720 en bf16. Seules les 16 couches
d'attention ont un cache KV ; les 48 couches GDN ont un état fixe, mais réservé par requête, d'où la
baisse avec 4 requêtes. Un test « aiguille dans une botte de foin » (code caché au milieu de 50 000 et
55 000 tokens) est réussi avec le cache fp8. Un prompt de 30 000 tokens se pré-remplit à ~1150-1275 t/s.

---

## 3. Ce qui a fonctionné (dans l'ordre chronologique)

### 3.1 Support du modèle dans le plugin GGUF (prérequis)
- Types GGML `Q2_0` (42), `PQ2_0` (142) et `PTQ1_0` (143), absents de `gguf` PyPI : `ggml_types.py`.
- Métadonnées `prism.hadamard.*` : `hadamard.py`. Les poids sont pliés : `W x = W_f · H(D·x)`.
  L'embedding est stocké en « latent » : `e = D·(H·l)`. Avec `gdn_v_grouped`, `ssm_out` est utilisé
  **tel quel**, sans la permutation tiled → grouped, tandis que les autres tenseurs GDN la gardent.
- Validation sur le vrai 27B, contre les poids bf16 de Qwen3.8-27B : corrélations de 0,885-0,89 avec
  les bonnes conventions, contre ~0,04 sans.
- 2B : les trois encodages donnent le même texte. La référence indépendante (déquantification numpy
  + dépliage) donne 96/96 premiers tokens identiques, avec |Δlogprob| moyen de 0,03.
- Correctif annexe : le plugin `main` ne s'importait plus avec ce vLLM (configs Qwen3.5 déplacées dans
  transformers).
- **Point de départ mesuré (PQ2_0, CUDA graphs)** : pp512 = 367 t/s, tg128 = 14,6 t/s. 85 % du temps
  de décodage était dans un GEMM Triton générique à ~100 Go/s.

### 3.2 Réorganisation des poids au chargement (sans perte)
- Dans les fichiers Q2_0 (blocs de 64), les échelles vont **par paires identiques** : la vraie
  granularité est de 128 poids. Les poids sont réécrits en codes 2 bits + une échelle fp16 par 128
  (2,125 bits/poids au lieu de 2,25), et le poids GGUF d'origine est libéré.
- Format « byte-plane » : dans un chunk de 512 poids, l'octet `w*4+b` contient, aux bits `2i`, le code
  du poids `i*128 + b*32 + w`. Un même chargement sert ainsi les deux chemins :
  `(mot >> 2i) & 0x03030303` donne 4 codes du groupe `i` pour `dot4`, et un décalage d'une tuile
  d'octets donne directement une tuile int8 pour le GEMM.
- Code : `quantization/ternary_fast.py` (`repack_ternary`), `csrc/ternary/ternary_kernels.cu`.

### 3.3 Quantification des activations : signes + Hadamard + int8 en un seul noyau
- Un noyau HIP applique `D`, la FWHT 1024 normalisée, puis quantifie en int8 par groupe de 128, dans
  l'ordre permuté attendu par les GEMV/GEMM. Il remplace le produit dense par la matrice de Hadamard.
- v1 (FWHT en LDS, 10 synchronisations) : ~14 µs par appel en décodage.
  **v2** (16 valeurs par thread, butterflies en registres et par shuffles, un seul échange LDS,
  écriture de 16 octets) : **~4,4 µs**. En vérification spéculative, 3,7 → 0,75 ms par pas.

### 3.4 GEMV de décodage en HIP avec `v_dot4` (`__builtin_amdgcn_sudot4`)
- Triton plafonnait à 100-134 Go/s : décoder 2 bits puis faire une FMA fp32 par poids coûte environ
  4 instructions par poids, et l'ALU sature.
- Activations en int8 + `dot4` (4 MAC par instruction) + extraction de 4 codes par `shift & mask` :
  **570-630 Go/s à M=1** (90 % de la bande passante crête), contre ~100 auparavant.
- Pour M=2..8 (vérification spéculative) : un wave traite R lignes et partage les chargements
  d'activations. Les codes sont extraits une fois par ligne, hors de la boucle sur M. Valeurs retenues :
  R=1 (M=1), 2 (M=2), 2 ou 4 (M=4, 4 pour N ≥ 65536). À M=4 : 390-560 Go/s selon la forme.

### 3.5 GEMM de prefill en int8 sur les unités matricielles RDNA4
- Triton int8 `tl.dot` sur le format byte-plane (un `dot` de K=128 par groupe, `num_warps=8`,
  `num_stages=1`) : ~105 TOPS. pp512 : 367 → **1648 t/s**.
- **Noyau HIP WMMA gfx12** (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`). La disposition des
  opérandes a été vérifiée expérimentalement. Tuiles 128×128, 8 waves (2×4), activations d'un groupe et
  octets de poids d'un chunk en LDS, préchargement du groupe suivant en registres pendant le calcul.
  Les codes signés sont obtenus sans retenue par `((v|0x80808080)-0x01010101)^0x80808080`.
  **118-137 TOPS**, soit +15 à 30 % sur Triton. Triton reste le repli hors gfx1200/1201.
- Version « petits M » (9 ≤ M ≤ 64) : un wave par tuile 16×16. Entrelacer les 4 groupes en 4
  accumulateurs indépendants raccourcit la chaîne de dépendance WMMA (32 → 8) : **+15-30 % à M=8**.

### 3.6 Autres gains de prefill
- Sous ROCm, pour les autres types GGUF à M ≥ 256 (la tête MTP en Q8_0) : déquantification puis
  hipBLASLt au lieu du MMQ CUDA porté, 2× plus rapide à M=512.
- Backend d'attention `TRITON_ATTN` au lieu de `ROCM_ATTN` : prefill 1545 → 1694 t/s, et décodage
  spéculatif sur long contexte 50 → 96 t/s (`ROCM_ATTN` utilise un noyau prefix-prefill lent pour les
  requêtes de plusieurs tokens).
- **Résultat : pp512 ≈ 2050 t/s** (TTFT ≈ 250 ms).

### 3.7 Décodage spéculatif MTP (étape intermédiaire, 57 → 106,6 t/s)
- Sans spéculation, le plafond physique du Q2_0 est d'environ 85 t/s : ~7 Go de poids par token à
  640 Go/s. **Une RTX 5090 atteint 130 t/s avec 1792 Go/s ; une RX 9070 ne le peut pas sans spéculation.**
- La tête MTP de Qwen3.8-27B (`mtp.*`, lue par requêtes HTTP de plage) est greffée dans une copie du GGUF
  sous forme de bloc `blk.64.nextn.*` (Q8_0) : `tools/graft_mtp.py`.
- Correctifs du plugin : le brouillon MTP lisait sa config depuis le chemin du GGUF.
- k=3 : acceptation ≈ 2,7 sur texte naturel, **96-98 t/s**.
- **Auto-distillation de la tête MTP** : 2000 prompts Alpaca/CodeAlpaca, réponses greedy de Bonsai
  (620 000 tokens), états cachés capturés par un hook sur la normalisation finale, entraînement PyTorch
  déroulé sur 3 étapes, comme à l'inférence. Acceptation estimée 2,55 → 2,86 ; serveur 97,8 → **106,6 t/s**.

### 3.8 DFlash2 (la configuration retenue)
- Drafter `naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2` (Apache-2.0), dérivé de
  `z-lab/Qwen3.8-27B-DFlash2` et déjà réentraîné sur les sorties de Bonsai. Il propose un bloc de
  8 tokens en une seule passe, en s'appuyant sur 5 couches de la cible (5, 19, 33, 47, 61).
- Correctifs nécessaires :
  - plugin : un brouillon non GGUF garde `config_format=auto` et `draft_load_config=auto`, sinon il était
    lu comme du GGUF ;
  - option `VLLM_GGUF_DRAFT_VOCAB_ON_CPU=1` : le brouillon allouait temporairement 2 × 2,37 Gio de tables
    de vocabulaire, ce qui ne tenait pas dans 16 Go ;
  - copie locale `~/.cache/dflash2-bonsai-nakli` avec `candidate_selector.{predecessor,successor}_codebook`
    renommés (le suffixe `.weight` venait de l'export MLX).
- Drafter en fp8 (`"quantization":"fp8"`) : 140,7 contre 133,4 t/s en bf16.
- Acceptation : 3,3 à 4,8 tokens par cycle selon le texte (contre 2,7-3,0 avec MTP).

### 3.9 La découverte décisive : `GPU_MAX_HW_QUEUES=1`
- Symptôme : performances bimodales et non reproductibles. Avec la même configuration, DFlash donnait
  68 ou 113 t/s, et le décodage simple 30 ou 57 t/s. Un cycle prenait 48 ms au lieu de 29, alors que
  le moteur passait 69 % du temps à attendre le GPU et que les fréquences étaient au maximum.
- Cause : le processus API de `vllm serve` ouvre lui aussi un contexte HIP, et chaque processus réserve
  4 files matérielles par défaut. Le GPU les sur-souscrit et alterne entre elles, selon l'ordre de création.
- `GPU_MAX_HW_QUEUES=1` : **113,4 / 140,7 t/s, reproductible**.
- `--async-scheduling` est passé explicitement avec DFlash.

### 3.10 Quatre requêtes simultanées, même pour un seul utilisateur
- `--max-num-seqs 4` donne 140,7 t/s en chat, contre 129,7 avec `--max-num-seqs 1`, à contexte égal
  (probablement grâce aux CUDA graphs capturés pour plusieurs tailles de lot). En contrepartie,
  4 requêtes réservent 4 états GDN, ce qui coûte ~6 000 tokens de contexte en fp8.

### 3.11 Cache KV fp8
- Double le contexte : 63 488 au lieu de 30 720 avec 4 requêtes. Coût : environ −4 % en génération
  (129 contre 134 t/s). Qualité vérifiée par le test « aiguille » à 50 000 et 55 000 tokens.

### 3.12 Plusieurs requêtes : `--prefill-schedule-interval 32` (2026-11-04)
- Symptôme : une requête qui génère tombe à ~2-13 t/s dès qu'une autre arrive avec un long prompt.
  Avec le chunked prefill (budget de 2048 tokens par pas), chaque pas de décodage embarque un bloc de
  prefill de ~1,2 s : la requête en cours n'avance que d'un cycle MTP par seconde, tant que dure le prefill.
- `--long-prefill-token-threshold 512` : les deux prefills avancent en parallèle et le TTFT de la première
  requête double (19 → 35 s). Ce n'est pas mieux.
- `--prefill-schedule-interval N` : tant qu'une requête génère, le prefill n'a droit qu'à un pas sur N.
  Une requête seule n'est pas touchée (pp identique).

| 2 requêtes, prompts de 25k (MTP, fp8) | tg de la requête en cours pendant le prefill de l'autre | TTFT de la 2ᵉ |
|---|---|---|
| défaut (1) | 12,8 t/s | 38,3 s |
| 16 | 25,4 t/s | 41,7 s |
| **32** (retenu, `--prefill-interval`) | **41,6 t/s** | 41,6 s |

- Avec 4 requêtes arrivées presque ensemble, rien ne change : vLLM lève le frein tant que la file
  d'attente ne se vide pas.

### Prefill selon la longueur du prompt (MTP, fp8, requête seule)
| tokens | 512 | 2048 | 8192 | 16384 | 32768 | 65536 |
|---|---|---|---|---|---|---|
| t/s | 2084 | 1908 | 1698 | 1477 | 1169 | 821 |

La baisse vient de l'attention (`TRITON_ATTN`) sur long contexte : de 32k à 64k, le prefill marginal tombe à ~630 t/s.

---

## 4. Ce qui n'a pas fonctionné (ou n'a rien apporté)

| Essai | Résultat | Pourquoi / remarque |
|---|---|---|
| GEMV Triton (formats « plane », reshape 3D) | 100-134 Go/s | limité par l'ALU (décodage + FMA par poids) |
| GEMM Triton avec décodage par « gather » | 35 TOPS | 16 lectures redondantes par mot + conversions de layout |
| GEMM Triton plane32 / byte-plane déroulé par 4 | 17-44 TOPS | `dot` à K=32, débordements de registres |
| Options AMD de Triton (`waves_per_eu`, `kpack`, `matrix_instr_nonkdim`) | aucun gain | plateau ~105 TOPS |
| WMMA « grandes tuiles » sans LDS | 21-85 TOPS | débordements de registres |
| WMMA petits M à M ≤ 4 | ≤ dot4 | le GEMV `dot4` reste meilleur jusqu'à M=4 |
| Pipelining logiciel du GEMV `dot4` | pire | pression sur les registres |
| Split-K entre waves dans le GEMV | pire (−10 à −25 %) | ce n'est pas le parallélisme qui limite |
| WMMA petits M : préchargement à 2 chunks, 2 tuiles par wave, 8 waves, activations en LDS | toutes pires | la latence mémoire de ce découpage reste la limite |
| WMMA « dalles de poids » en LDS (lecture façon GEMV) | 272-362 contre 265-557 Go/s | synchronisations, petits blocs |
| WMMA split-K avec réduction par le dernier bloc | pire (−10 à −30 %) | idem ; code gardé mais désactivé (`TERNARY_WMMA_SPLIT`) |
| Tête MTP en Q4_0 au lieu de Q8_0 | 93,1 contre 96,4 t/s | la tête MTP n'était pas le goulot |
| MTP avec k=4 / k=5 | 62 / 94 t/s | moins bien que k=3 |
| Auto-distillation MTP avec 2× plus de données | 2,857 → 2,867 | plafond atteint ; l'entraînement saturait la VRAM et calait après l'évaluation |
| Vocabulaire réduit pour le brouillon (ids BPE < N) | rejeté | couverture de 91-94 % en anglais et code, 64 % en français |
| `extract_hidden_states` de vLLM | erreur | « Duplicate layer name » sur ce modèle hybride ; remplacé par un hook sur la norme finale |
| Hook sur le module backbone | jamais appelé | `support_torch_compile` remplace `__call__` |
| DFlash k=3 / k=5 | 110,7 / 125,9 t/s en chat | k=7 (bloc complet) est le meilleur |
| Drafter DFlash2 de ProCreations | 136,5 t/s en chat | équivalent, légèrement moins bon |
| Drafter DFlash2 en bf16 | 133,4 t/s en chat | fp8 meilleur (140,7) |
| `HIP_FORCE_DEV_KERNARG=1`, `DEBUG_HIP_GRAPH_PACKET_CAPTURE=1` | aucun gain | |
| Profil d'énergie `COMPUTE` (sudo) | inutile | la carte est déjà au max pendant la génération |
| Vider le cache de compilation vLLM, changer `gpu-memory-utilization` | aucun effet | fausses pistes pendant la recherche de la bimodalité |
| Cache KV TurboQuant `turboquant_4bit_nc` / `turboquant_k3v4_nc` | **57 344** tokens max (moins que fp8 : 64 272) | il faut 2 contournements pour qu'il démarre (voir §4.1), et la plupart des couches restent en précision native |
| Cache KV TurboQuant `turboquant_3bit_nc` | 73 728 tokens, mais **texte dégénéré** (« a a a a… ») et OOM sur un prompt de 62 000 tokens | inutilisable sur cette carte |
| Cache KV UltraQuant `ultraquant_4bit` | refusé | « No KV cache layout satisfies every supported set » (modèle hybride + drafter) |
| Cache KV Octave (`octave_k3v4`…) | refusé | backend `ROCM_OCTAVE` non supporté sur gfx1201 |


### 4.1 Détail des essais de cache KV compressé (TurboQuant, UltraQuant, Octave)
« HyperQuant » n'existe pas dans vLLM ; les formats de cache KV compressé disponibles sont TurboQuant,
UltraQuant et Octave (`--kv-cache-dtype`). Ils ne réduisent que le cache des 16 couches d'attention :
ils n'apportent que du contexte, pas de vitesse.
1. Premier refus : l'attention du drafter DFlash utilise une fenêtre glissante et n'est pas causale, deux
   cas que TurboQuant ne gère pas. Contournement : `--kv-cache-dtype-skip-layers sliding_window`.
2. Deuxième refus : le préremplissage de TurboQuant appelle FlashAttention (CK), dont la roue ROCm n'a pas
   de noyaux pour gfx1201 (« device kernel image is invalid »). Contournement : masquer `flash_attn`
   avec un module qui lève `ModuleNotFoundError`, ce qui fait basculer TurboQuant sur son repli SDPA.
3. Résultats : 4 bits et k3v4 donnent moins de contexte que fp8. 3 bits donne +15 % de contexte, mais
   un texte dégénéré dès le début de la réponse, et le repli SDPA déborde de mémoire sur un prompt de
   62 000 tokens. Le débit affiché (208 t/s en chat) était un artefact : un texte répétitif est trivial
   à deviner pour le drafter.

**Conclusion : le cache KV fp8 reste le meilleur choix sur cette carte.**
---

## 5. Pièges de mesure rencontrés

- **Infinity Cache de 64 Mo** : un micro-benchmark qui relit la même matrice donne plus de 1000 Go/s.
  Il faut faire tourner plusieurs copies (plus de 400 Mo) pour mesurer la VRAM.
- **`LLM(...)` en multiprocessus** : 28 t/s, contre 59 dans le même processus et 57 via `vllm serve`.
  Mesurer via `vllm serve`.
- **Files matérielles** (§3.9) : sans `GPU_MAX_HW_QUEUES=1`, toutes les mesures en `serve` sont aléatoires.
- Le **profileur torch** gonfle les trous entre noyaux. `rocprofv3 --kernel-trace` (dans le SDK ROCm
  du venv) donne la vraie chronologie.
- Le **décodage spéculatif dépend du texte** : sur 3 prompts, de 87 à 200 t/s. La complétion brute
  d'une consigne (sans template de chat) donne des débits plus bas que le chat.
- **Sorties pas identiques au bit près** avec la spéculation : la vérification par lots de 8 (WMMA)
  n'additionne pas dans le même ordre que le décodage token par token. On observe des divergences
  tardives sur des quasi-égalités (après ~600-800 caractères ; par exemple `left` contre `left_half`),
  avec un texte qui reste cohérent.
- Sous zsh, `$VAR` contenant des espaces n'est pas découpé en arguments (il faut `${=VAR}`).

---

## 6. Pourquoi 130 t/s n'est pas atteint sur de la prose longue

Un cycle DFlash (chiffres de `rocprofv3`) : **~31,6 ms**, dont 26,2 ms de calcul GPU :
- vérification à M=8 sur les 64 couches et le lm_head : **16,1 ms** (≈ 440 Go/s pour 7,1 Go) ;
- drafter fp8 : ≈ 4,4 ms ; GDN, attention, normes : ≈ 3 ms ; `act_quant` : 0,8 ms ;
- ≈ 5,4 ms de trous entre ~1375 noyaux (≈ 4 µs par nœud de graphe HIP).

Le drafter accepte ~5 tokens par cycle sur du code, mais ~3 sur de la prose libre : 3 / 31,6 ms ≈ 95 t/s.
Pour 130 t/s, il faudrait ≈ 23 ms par cycle, donc une vérification proche de la bande passante crête,
que huit variantes du noyau n'ont pas atteinte. La RTX 5090 citée par PrismML a 2,8× plus de bande passante.

Pistes non réalisées :
1. Un noyau de vérification à M=8 entièrement repensé : gain incertain, de l'ordre de 7 à 12 %.
2. Un drafter réentraîné sur de la prose : impossible en entier sur 16 Go (1,9 milliard de paramètres
   avec Adam ≈ 30 Go). Un réglage partiel (`fc`/norme) apporte ~3 % selon l'auteur du drafter.
3. Un format plus dense (1,6 bit/poids, base 3) : environ 19 % d'octets en moins, mais un décodage plus
   coûteux en calcul (PrismML le juge perdant sur les GPU riches en calcul).

---

## 7. Fichiers

### Plugin (`~/Desktop/Perso/vllm-gguf-plugin`, branche `bonsai-ternary-hadamard`, non commité)
| Fichier | Rôle |
|---|---|
| `vllm_gguf_plugin/ggml_types.py` | enregistre les types Q2_0 / PQ2_0 / PTQ1_0 |
| `vllm_gguf_plugin/hadamard.py`, `quantization/hadamard_methods.py` | métadonnées `prism.hadamard`, transformations |
| `vllm_gguf_plugin/quantization/ternary_fast.py` | réorganisation des poids, custom op `gguf_ternary_linear`, choix GEMV / WMMA / Triton |
| `vllm_gguf_plugin/csrc/ternary/ternary_kernels.cu` | noyaux HIP : `act_quant`, GEMV `dot4`, WMMA (grand, petit, split-K désactivé) |
| `vllm_gguf_plugin/triton/gemm/ternary_quant/packed.py` | GEMM Triton int8 (repli hors gfx12) |
| `vllm_gguf_plugin/quantization/linear.py` | branchement du chemin rapide, déquantification ROCm pour M ≥ 256 |
| `vllm_gguf_plugin/plugin.py` | config / poids des brouillons MTP et DFlash, option `VLLM_GGUF_DRAFT_VOCAB_ON_CPU` |
| `vllm_gguf_plugin/weights_adapter/qwen3_5.py`, `loader.py`, `quantization/config.py` | conventions GDN, plan de chargement Hadamard |
| `tests/test_ternary_hadamard.py` (+ `test_gguf_generation.py`) | 104 tests (152 avec les autres suites) |

Variables d'environnement de réglage (non nécessaires en usage normal) : `TERNARY_GEMV_ROWS`,
`TERNARY_GEMV_SPLIT`, `TERNARY_WMMA_SMALL`, `TERNARY_WMMA_SPLIT`.

### Ce dossier (`~/Desktop/Perso/bonsai-serve`)
- `bonsai` : la commande (liée dans `~/.local/bin/bonsai`) ; `Opti.md` : ce document.
- `tools/` : scripts de mesure et de préparation (`graft_mtp.py`, `distill_*.py`, `serve_bench.py`,
  `serve_chat.py`, `longctx.py`, `cycles.py`, prototypes de noyaux...).

### Données (`~/.cache`)
| Chemin | Taille | Nécessaire ? |
|---|---|---|
| `dflash2-bonsai-nakli/` | 3,8 Go | **oui** (drafter) |
| `bonsai-27b-mtp-tuned/` | 8 Go | seulement pour `--draft mtp` |
| `bonsai-27b-mtp/`, `bonsai-27b-mtp-q4/` | 2 × 8 Go | non (essais) |
| `bonsai-distill/` | ~23 Go | non (données d'auto-distillation) |
| `bonsai-ref/` | 3,8 Go | non (référence F16 du 2B) |
| `dflash2-bonsai-procreations/` | 3,8 Go | non (drafter comparé) |
| `vllm/torch_compile_cache.bak_*` | 5,2 Go | non |

---

## 8. Reproduire

```bash
# Plugin (noyaux HIP pour gfx1201 uniquement ; librocthrust-dev requis)
cd ~/Desktop/Perso/vllm-gguf-plugin
PYTORCH_ROCM_ARCH=gfx1201 uv pip install --python ~/Desktop/Perso/vllm/.venv/bin/python -e . --no-build-isolation

# Drafter DFlash2 avec les deux tenseurs renommés
~/Desktop/Perso/vllm/.venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
import os, shutil
src = snapshot_download("naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2", allow_patterns=["config.json", "model.safetensors"])
dst = os.path.expanduser("~/.cache/dflash2-bonsai-nakli"); os.makedirs(dst, exist_ok=True)
shutil.copy(f"{src}/config.json", dst)
sd = load_file(f"{src}/model.safetensors")
for k in ("candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook"):
    sd[k] = sd.pop(k + ".weight")
save_file(sd, f"{dst}/model.safetensors", metadata={"format": "pt"})
EOF

# (optionnel) GGUF avec tête MTP greffée : tools/graft_mtp.py <Q2_0.gguf> <sortie.gguf> Q8_0
# (MTP_LOCAL=mtp_tuned.safetensors pour la tête réentraînée, voir tools/distill_*.py)

bonsai launch --detach
```

Tests : `cd ~/Desktop/Perso/vllm-gguf-plugin && ~/Desktop/Perso/vllm/.venv/bin/python -m pytest tests/test_ternary_hadamard.py -q`
(lancer pytest depuis le dossier du plugin, jamais depuis `~/Desktop/Perso/vllm`, dont les sources masquent la roue installée).

---

## 9. Profil « 4 requêtes × 100k » (`--kv int4`, par défaut depuis le 2026-11-04)

Objectif : 4 requêtes simultanées de 100 000 tokens de contexte chacune, avec ≥ 600 t/s de prefill
et ≥ 50 t/s de génération par requête.

### 9.1 Le problème de mémoire
Pour 4 × 100k, le cache KV fp8 (16 couches d'attention + la couche MTP, 4 têtes KV × 256) prend
13,6 Go : impossible sur 16 Go avec 7,5 Gio de poids. Il restait ~6,6 Go pour le KV et les états GDN.

| Étape | Requêtes de 100k qui tiennent |
|---|---|
| fp8 (avant) | 1,05 |
| int4 de vLLM (Triton) | 2,31 — mais pp 230 t/s et tg 9,6 t/s à 25k : inutilisable |
| + état GDN bf16, 0,975 de mémoire | 2,88 |
| + RecoverSSM pour GDN (une seule copie d'état par requête) | 3,43 |
| + V en 3 bits (K4/V3, 232 o par token et tête) | 3,72 |
| + `--kv-cache-memory` fixé (le profil de vLLM surestime activations et graphes) | **4,00** |

### 9.2 Cache KV K4/V3 et noyaux HIP (plugin `bonsai_int4_attn.py`, `csrc/int4attn/int4_attn.hip`)
- Rotation de Hadamard aléatoire orthonormée (signes + Sylvester 256) avant quantification, comme
  QuaRot / TurboQuant : les valeurs deviennent quasi gaussiennes, sans canal aberrant.
- K : 4 bits asymétriques, une échelle fp32 par (token, tête), point zéro dans les bits bas de la mantisse.
- V : 3 bits, seuillage à moyenne ± 2,15σ (quantificateur uniforme quasi optimal pour une gaussienne),
  stocké **transposé par tuile de 16 tokens** en deux plans de bits, pour que l'opérande WMMA se charge
  en 48 octets contigus par lane, sans LDS.
- Astuce clé : on calcule Sᵀ = K·Qᵀ (WMMA int8, Q quantifié int8) ; l'accumulateur a alors exactement la
  disposition de l'opérande B de Oᵀ = Vᵀ·Pᵀ (WMMA fp16) : P ne passe jamais par la mémoire partagée.
- Décodage / vérification : split-K à grille fixe (compatible CUDA graphs), ~0,97 ms par couche pour
  4 × 100k (limite mémoire ~0,62 ms). Prefill : ~58 TFLOPS effectifs.
- Qualité (NLL moyenne sur 2 × 12 000 tokens de code, par segments [0-2k] / [2k-6k] / [6k-12k]) :

| Cache | NLL |
|---|---|
| fp8 de vLLM (référence) | 1,4514 / 1,2766 / 1,1402 |
| int4 min/max (K4V4) | 1,4528 / 1,2812 / 1,1429 |
| **K4 + V3 seuillé (retenu)** | 1,4530 / 1,2783 / 1,1412 |
| K3 + V3 seuillés | 1,4606 / 1,2893 / 1,1486 (+0,6 à 1 %) |

### 9.3 RecoverSSM pour Gated DeltaNet (plugin `bonsai_gdn_recoverssm.py`)
vLLM garde, en mode « align » (cache de préfixe), 2 + k copies de l'état GDN par requête et par
groupe (k = 3 tokens MTP) : 0,84 Go par requête. vLLM implémente RecoverSSM pour KDA (Kimi) : la
vérification part d'un seul point de contrôle et enregistre les corrections par token ; après
l'échantillonnage, les tokens acceptés sont repliés dans l'état. GDN = KDA avec une porte scalaire par
tête et des clés GQA : les noyaux de validation KDA sont réutilisés tels quels, avec un noyau de
vérification natif pour GDN (2,1 ms par cycle pour 48 couches, contre 3,0 pour le noyau d'origine).
Pièges rencontrés : le chemin MTP fusionné de GDN doit être désactivé ; la largeur de la table d'états
fixe la fenêtre de la convolution (à garder à 1 + k) ; `num_accepted_tokens` du runner ne doit jamais
être modifié en place ; MRV2 recopie la métadonnée GDN entre groupes via `update_block_table`.
Nécessite `VLLM_USE_V2_MODEL_RUNNER=1`.

### 9.4 Résultats (RX 9070, prompts de code réels de ~95k tokens, aiguille au milieu)
| Mesure | Valeur |
|---|---|
| 4 requêtes de ~95k en même temps dans la VRAM | oui, sans préemption, 4 aiguilles retrouvées |
| Prefill d'un prompt de 95k (requête seule) | 1080-1089 t/s |
| Prefill d'un prompt de 95k pendant que 3 autres décodent (intervalle 6) | **634 t/s** (les 3 autres gardent ~24 t/s) |
| Génération, 4 requêtes de ~95k qui décodent ensemble | **51,6 / 57,7 / 52,8 / 59,2 t/s** |
| Génération, 1 requête à 95k | ~105 t/s |
| Génération, contexte court : 1 requête / 4 requêtes | 80-108 t/s / ~185 t/s au total |

Compromis prefill / génération quand un prompt long arrive pendant que 3 requêtes génèrent
(`--prefill-interval N`) : N=1 → prefill 1008 t/s, les autres ~6 t/s ; N=6 → 634 t/s et ~24 t/s ;
N=8 → 552 t/s et ~25 t/s. **Avoir à la fois ≥ 600 t/s de prefill et 3 × 50 t/s de génération à ~100k
de contexte dépasse la capacité de la carte** : à cette profondeur un token de prefill coûte ~1,05 ms
(dont ~0,65 ms d'attention), et 3 requêtes à 50 t/s occupent ~70 % du GPU.

### 9.5 Essais sans gain
| Essai | Résultat |
|---|---|
| `--long-prefill-token-threshold 512` | les deux prefills avancent ensemble ; le premier TTFT double |
| Prefill avec tuiles K/V partagées en LDS (4 waves) | 2× plus lent (30 TFLOPS) : 36 Ko de LDS par groupe, occupation trop faible |
| Tête MTP en Q4_0 | −0,18 Gio mais plus lent (159 contre 186 t/s à 4 requêtes) : noyaux Q4 moins efficaces |
| `--max-num-batched-tokens 1024` + moins de graphes | estimation de capacité de vLLM *en baisse* (variance du « non-torch ») |

### 9.6 Pistes restantes (mesurées)
- GEMM ternaire à M=16 (4 requêtes × 4 tokens de vérification) : 335 Go/s alors que le motif d'accès
  atteint 516 Go/s (586 pour `down`) ; ~18 ms par cycle, ~6 ms récupérables.
- Tête MTP Q8_0 : ~7 ms par cycle à ~240-450 Go/s.
- Attention de décodage : 0,97 ms par couche contre ~0,62 ms à la limite mémoire.
- Attention de prefill : 58 TFLOPS sur ~178 possibles (WMMA int8 + fp16) ; PV en fp8 à évaluer.

### 9.7 Fichiers
- `bonsai` : profil `--kv int4` (variables `BONSAI_INT4_ATTN`, `BONSAI_GDN_RECOVERSSM`,
  `VLLM_USE_V2_MODEL_RUNNER`, cache KV dimensionné exactement, lots de 512 tokens, intervalle 6).
- Plugin : `bonsai_int4_attn.py`, `csrc/int4attn/int4_attn.hip` (compilé à la volée dans
  `~/.cache/bonsai-int4attn*`), `bonsai_gdn_recoverssm.py` ; branchés dans `plugin.py` par variables
  d'environnement. Sauvegarde du plugin d'avant : `~/Desktop/Perso/backups/`.

### 9.8 Suite (2026-11-05) : noyaux de décodage et scénario mixte
Nouveaux noyaux (dans `csrc/int4attn/int4_attn.hip`, activés par défaut avec `--kv int4`) :
- **GEMM ternaire M ≤ 16** (`tern_m16`, dans `ternary_fast._ternary_linear`, `BONSAI_TERN16=1`) : K découpé entre
  2 ou 4 waves (réduction en LDS), activations chargées par 16 o, échelles d'activation en LDS : **~485 Go/s
  contre ~340** pour `wmma_small` à M=16 (vérification spéculative de 4 requêtes).
- **GEMM ternaire 16 < M ≤ 64** (`tern_m64`, 2 tuiles M par wave) : 1,6-1,9× plus rapide que `wmma_small`.
- **Tête MTP en int8 tourné** (`bonsai_mtp_w8.py`, `BONSAI_MTP_W8=1`) : à la charge, les poids Q8_0 sont tournés par la
  même Hadamard signée que le quantificateur d'activations (W' = W·Tᵀ) et requantifiés par groupe de 128 ; le
  noyau `w8_m16` les lit directement comme opérande WMMA : 3-6× plus rapide que MMVQ/MMQ à M ≤ 64.
- Effet : 4 requêtes courtes **186 → 224 t/s** au total ; 4 requêtes de **~99k : 58-62 t/s chacune**.
- Le cache `torch.compile` de vLLM ne voit pas ces patches : le profil int4 utilise son propre
  `VLLM_CACHE_ROOT` (`~/.cache/vllm-bonsai-int4`). Après une modification des patches, le vider.

Scénario mixte (3 requêtes ~95k qui génèrent + 1 prompt de 99k qui arrive) :

| Réglage | prefill du nouveau prompt | génération des 3 autres pendant ce temps |
|---|---|---|
| intervalle 1 | ~1000 t/s | ~6-9 t/s |
| intervalle 6 (défaut) | 630-686 t/s | ~22-26 t/s |
| `--prefill-chunk 48` (prefill mêlé à chaque pas) | ~510 t/s | ~28 t/s |
| `--prefill-chunk 48 --no-async` | 333 t/s | 18 t/s (coût CPU non masqué) |
| objectif | 600 t/s | 50 t/s |

Pourquoi l'objectif simultané n'est pas atteint : un pas mixte (12 tokens de vérification + 48 de prefill) dure
~90 ms au lieu de ~55 : GEMM à M=60 ~36 ms (~70 TOPS, pic int8 265), attention du morceau ~17 ms en moyenne
(33 ms à 99k de profondeur, ~60 TFLOPS), attention des décodeurs ~13 ms (sur un flux parallèle), GDN/MTP/reste
~20 ms. Il faudrait un GEMM proche du pic int8 à M≈60 et une attention ~1,7× plus rapide.
Essais sans gain : grille du noyau split-K réordonnée pour la L2 (aucun effet : le morceau est limité par le calcul) ;
k=4/5 tokens MTP (manque de mémoire : allocation de 340 Mo refusée).
Profil réel d'un pas mixte (3 décodeurs + morceau de 48 tokens, `rocprofv3`) : `tern_m64` (GEMM à M≈60) ≈ 32 ms
sur ~66 ms par pas. Essais pour l'accélérer, sans gain : tuiles de 48/64 lignes du GEMM du plugin (`tern_tile`,
≤ +10 %), activations partagées en LDS entre 8 waves (`tern_lds`, aucun gain). Le GEMM à petit M reste vers
70-95 TOPS (pic int8 265) : c'est le verrou principal du scénario mixte.
Aussi sans gain : préchargement logiciel des activations d'un groupe à l'autre (`tern_m64p`, 0,72 contre 0,68 ms à
M=60). Ni le trafic d'activations, ni leur latence, ni la forme des tuiles n'expliquent le plafond (~35 % du pic
WMMA) ; sans compteurs matériels (rocprofv3-avail ne fonctionne pas ici), la cause reste à trouver.
Compteurs matériels : `rocprofv3 --pmc` échoue (« aqlprofile API table load failed », puis SIGFPE avec la version
du venv strata). Probablement des droits (root) ou gfx1201 non pris en charge par cette version d'aqlprofile.

## 10. Charge agentique (F1)

Outil : `serve/tools/agent_bench.py`. 4 agents aux historiques distincts, départ à 90k tokens, +2-4k tokens par tour,
256 tokens de sortie, `--sync-turns` (barrière : tous les agents démarrent le tour N ensemble), `--max-context 100000`.
Il affiche pour chaque requête les tokens `cached/prompt` (nécessite le flag serveur `--enable-prompt-tokens-details`,
désormais dans le lanceur).

### 10.1 Référence (sans synchronisation)
- Tour 1 (froid, 4 × 90k) : prefill moyen 564 t/s, décodage 15 t/s, TTFT 206 s.
- Tours suivants (chauds) : delta prefill 91-360 t/s, décodage moyen 15-53 t/s, minimum 9-21 t/s.
- Taux de hit du cache global : 72 % (tour froid inclus).

### 10.2 Diagnostic
- Hit du préfixe sur les tours chauds : 94-97 % par requête (cache de préfixes et mode d'alignement Mamba
  fonctionnent avec MTP, RecoverSSM et int4).
- Tokens recalculés en plus par tour : 0-1840 (bloc de checkpoint GDN de 1840) + la réponse précédente (~256).
- Pas d'éviction (pool de 400k tokens).
- TTFT restant de 6-27 s : file d'attente. ~13k nouveaux tokens par tour pour 4 agents, préfillés pendant que
  les autres décodent (intervalle de prefill 6).

### 10.3 Conclusion
Le décodage seul fait ~58-62 t/s à 100k : il n'y a pas de marge. Tout prefill concurrent fait passer le décodage
sous 60 t/s. Suite : accélérer le décodage (F3 Windowed-MTP, F4 noyaux d'attention HIP), puis F2 (découpage du
prefill selon une échéance) pour partager la marge.

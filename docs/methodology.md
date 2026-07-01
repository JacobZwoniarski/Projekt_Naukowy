# Dokumentacja i część metodyczna raportu

## 1. Cel projektu

Projekt jest laptopową reprodukcją głównej idei pracy Radford et al. (2021), czyli uczenia wspólnej przestrzeni reprezentacji obrazów i tekstów metodą kontrastową. Zamiast pełnoskalowego CLIP-a trenowanego na bardzo dużym zbiorze par obraz-tekst, implementacja używa mniejszego modelu Mini-CLIP oraz zbioru Flickr8k. Celem nie jest osiągnięcie wyników oryginalnego CLIP-a, tylko sprawdzenie, czy ta sama procedura uczenia pozwala uzyskać mierzalne dopasowanie obrazów i podpisów w ograniczonych warunkach obliczeniowych.

Projekt obejmuje trzy główne elementy:

- przygotowanie danych obraz-tekst,
- trening dwuwieżowego modelu obrazowego i tekstowego,
- ewaluację przez wyszukiwanie obraz-tekst oraz prosty transfer zero-shot.

## 2. Dane

Podstawowym zbiorem danych jest `jxie/flickr8k` ładowany przez bibliotekę Hugging Face `datasets`. Każdy przykład zawiera obraz oraz do pięciu podpisów tekstowych (`caption_0` ... `caption_4`). Implementacja dzieli dane na zbiory `train`, `validation` i `test`; jeżeli zbiór nie ma klucza `validation`, używany jest klucz `dev`.

W treningu każdy obraz jest łączony z jednym podpisem. Podpis jest wybierany deterministycznie zależnie od epoki i indeksu przykładu, dzięki czemu kolejne epoki wykorzystują różne podpisy tego samego obrazu bez losowego przetasowywania na poziomie pojedynczego przykładu.

Przetwarzanie obrazów:

- obrazy są konwertowane do RGB,
- w konfiguracji bazowej używany jest `Resize(image_size + 16)` i `CenterCrop(image_size)`,
- w konfiguracji `configs/flickr8k_strong.yaml` podczas treningu używana jest lekka augmentacja: `RandomResizedCrop` oraz `RandomHorizontalFlip`,
- normalizacja używa średnich i odchyleń standardowych stosowanych w CLIP.

Przetwarzanie tekstu:

- tokenizer jest prosty i oparty na wyrażeniu regularnym `[a-z0-9]+`,
- tekst jest sprowadzany do małych liter,
- słownik budowany jest wyłącznie na podpisach treningowych,
- sekwencja zawiera tokeny specjalne `<bos>`, `<eos>`, `<pad>` i `<unk>`,
- maksymalna długość sekwencji w aktualnych konfiguracjach wynosi 32 tokeny.

Dla testów technicznych bez pobierania Flickr8k dostępny jest także tryb syntetyczny, który generuje proste obrazy kolorowych kwadratów i odpowiadające im podpisy.

## 3. Model

Model ma architekturę dwuwieżową, analogiczną do CLIP:

- enkoder obrazu przekształca obraz w wektor cech,
- enkoder tekstu przekształca podpis w wektor cech,
- oba wektory są rzutowane do wspólnej przestrzeni o wymiarze `embed_dim`,
- reprezentacje są normalizowane do długości 1,
- podobieństwo obrazu i tekstu jest iloczynem skalarnym z uczoną skalą logitów.

Enkoder obrazu bazuje na `ResNet-18` z `torchvision`, trenowanym od zera (`weights=None`). Ostatnia warstwa klasyfikacyjna ResNeta jest usuwana, a jej miejsce zajmuje liniowa projekcja do przestrzeni wspólnej.

Enkoder tekstu składa się z:

- embeddingów tokenów,
- uczonych embeddingów pozycyjnych,
- małego `TransformerEncoder`,
- normalizacji warstwowej,
- liniowej projekcji do przestrzeni wspólnej.

Reprezentacją całego tekstu jest stan ukryty w pozycji tokenu `<eos>`. W konfiguracjach projektu tekstowy Transformer ma 2 warstwy, szerokość 256, 4 głowy uwagi i wymiar warstwy feed-forward 512.

## 4. Funkcja celu

Trening używa symetrycznej straty kontrastowej CLIP. Dla batcha złożonego z `N` dopasowanych par obraz-podpis model wyznacza macierz podobieństw `N x N`. Element na przekątnej odpowiada poprawnej parze, a pozostałe elementy są negatywnymi przykładami wewnątrz batcha.

Strata jest średnią z dwóch kierunków:

- klasyfikacji poprawnego tekstu dla obrazu,
- klasyfikacji poprawnego obrazu dla tekstu.

Formalnie implementacja liczy `cross_entropy(logits_per_image, targets)` i `cross_entropy(logits_per_text, targets)`, gdzie `targets = [0, 1, ..., N-1]`, a końcowa strata to średnia obu wartości.

## 5. Procedura treningu

Trening jest konfigurowany plikami YAML w katalogu `configs/`. Konfiguracja bazowa `configs/flickr8k.yaml` jest szybkim wariantem startowym, a `configs/flickr8k_strong.yaml` jest wariantem rekomendowanym do pełniejszego eksperymentu.

Najważniejsze parametry konfiguracji `flickr8k_strong`:

- rozmiar obrazu: 160,
- batch size: 128,
- liczba epok: 50,
- optimizer: `AdamW`,
- learning rate: `0.0004`,
- weight decay: `0.01`,
- gradient clipping: `1.0`,
- warmup: 250 kroków,
- scheduler: warmup + cosine decay,
- monitorowana metryka walidacyjna: `mean_r@1`.

Podczas treningu zapisywane są:

- `config.yaml` z konfiguracją konkretnego uruchomienia,
- `vocab.json` ze słownikiem,
- `checkpoint_last.pt` z ostatnią epoką,
- `checkpoint_best.pt` z najlepszym wynikiem walidacyjnym,
- `metrics.json` z historią treningu, czasem działania, urządzeniem i wersjami pakietów.

Losowość jest kontrolowana przez `seed = 42`. Kod ustawia ziarna dla `random`, `numpy`, `torch` oraz, jeśli jest dostępne, dla CUDA. Dodatkowo wyłącza niedeterministyczny dobór algorytmów cuDNN przez `torch.backends.cudnn.benchmark = False` i `torch.backends.cudnn.deterministic = True`.

## 6. Ewaluacja

### 6.1 Retrieval obraz-tekst

Główna ewaluacja sprawdza, czy model potrafi odzyskać właściwy tekst dla obrazu oraz właściwy obraz dla tekstu. Dla zbioru walidacyjnego lub testowego kod osobno koduje wszystkie obrazy i wszystkie podpisy, a następnie wyznacza macierz podobieństw.

Raportowane są metryki:

- `Text R@1`, `Text R@5`, `Text R@10` - odsetek obrazów, dla których co najmniej jeden poprawny podpis znalazł się w top-k,
- `Image R@1`, `Image R@5`, `Image R@10` - odsetek podpisów, dla których poprawny obraz znalazł się w top-k,
- `mean_r@1` - średnia z `Text R@1` i `Image R@1`.

Artefakty ewaluacji retrieval są zapisywane jako:

- `retrieval_table.csv`,
- `retrieval_table.md`,
- `retrieval_metrics.json`.

### 6.2 Transfer zero-shot na CIFAR-10

Druga ewaluacja jest uproszczonym testem transferu zero-shot. Model nie jest dotrenowywany na CIFAR-10. Zamiast tego nazwy klas są zamieniane na prompty tekstowe, kodowane przez enkoder tekstu i porównywane z embeddingami obrazów CIFAR-10.

Porównywane są trzy warianty promptów:

- sama nazwa klasy,
- pojedynczy prompt `a photo of a/an {label}`,
- średnia z kilku promptów zdefiniowanych w konfiguracji.

Wyniki są zapisywane jako:

- `prompt_ablation.csv`,
- `prompt_ablation.json`,
- `prompt_ablation.png`.

Ten eksperyment należy traktować jako dodatkowy test generalizacji. Model jest trenowany od zera na małym zbiorze Flickr8k, więc wynik zero-shot na CIFAR-10 ma ograniczoną wartość porównawczą względem pełnego CLIP-a.

## 7. Wyniki referencyjne

Według aktualnego README najlepsza konfiguracja projektu to `configs/flickr8k_strong.yaml`. Dla testu Flickr8k z `checkpoint_last.pt` uzyskano:

| Split | Text R@1 | Text R@5 | Text R@10 | Image R@1 | Image R@5 | Image R@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flickr8k test | 4.10 | 14.90 | 20.50 | 3.84 | 14.02 | 21.52 |

Dla prompt ablation na CIFAR-10 z `checkpoint_best.pt` uzyskano:

| Wariant promptu | Accuracy |
| --- | ---: |
| Sama nazwa klasy | 13.70 |
| `a photo of a {label}` | 16.00 |
| Ensemble promptów | 14.20 |

Interpretacja: model uczy się użytecznego dopasowania obraz-podpis, ale transfer zero-shot pozostaje słaby. Jest to zgodne z ograniczeniami skali danych, rozmiaru modelu i treningu od zera.

## 8. Powtarzalność eksperymentu

Pełny rekomendowany przebieg:

```bash
uv run python -m miniclip_repro.train --config configs/flickr8k_strong.yaml --output-dir outputs/flickr8k-strong-160-b128
uv run python -m miniclip_repro.eval_retrieval --config configs/flickr8k_strong.yaml --checkpoint outputs/flickr8k-strong-160-b128/checkpoint_last.pt --output-dir outputs/flickr8k-strong-160-b128-last
uv run python -m miniclip_repro.eval_zeroshot --config configs/flickr8k_strong.yaml --checkpoint outputs/flickr8k-strong-160-b128/checkpoint_best.pt --output-dir outputs/flickr8k-strong-160-b128
```

Szybki test bez pełnych danych:

```bash
uv run python -m miniclip_repro.run_all --config configs/flickr8k.yaml --fast-dev-run --synthetic-data --skip-zeroshot
```

Testy jednostkowe:

```bash
uv run pytest
```

## 9. Ograniczenia

Najważniejsze ograniczenia metodyczne:

- Flickr8k jest niewielki w porównaniu ze skalą danych użytych w oryginalnym CLIP-ie,
- model jest trenowany od zera, bez wag ImageNet ani wag CLIP,
- tokenizer jest prosty i nie używa segmentacji BPE ani gotowego tokenizera CLIP,
- ResNet-18 i mały Transformer tekstowy mają dużo mniejszą pojemność niż modele z pracy źródłowej,
- ewaluacja zero-shot na CIFAR-10 używa ograniczonej liczby przykładów określonej w konfiguracji,
- wyniki mogą zależeć od urządzenia, wersji bibliotek i dostępnej precyzji obliczeń.

W raporcie końcowym wyniki należy więc interpretować jako reprodukcję mechanizmu i procedury eksperymentalnej, a nie jako bezpośrednie porównanie jakości z oryginalnym CLIP-em.

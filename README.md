# Automatický přepis řeči letecké komunikace do textu
## Bakalářská práce
Veronika Nevařilová, vedoucí: Ing. Igor Szőke, Ph.D.

Toto README slouží jako stručný popis struktury této práce a jako jednoduché vysvětlení účelů jednotlivých skriptů.

V repozitáři se nacházejí všechny použité skripty pro úpravu surových dat, vytvoření datasetů, trénovací skript i skripty pro evaluaci modelů. Zpracovávaná data (audio s přepisy) nejsou z důvodu jejich citlivé povahy připojeny.

Text bakalářské práce je k dispozici [zde](https://www.vut.cz/studenti/zav-prace/detail/150718).

Na Hugging Face byly zveřejněny nejlepší modely pro [plný](https://huggingface.co/BUT-FIT/whisper-ATC-czech-full) a pro [zkrácený](https://huggingface.co/BUT-FIT/whisper-ATC-czech-short) přepis.


Pro nainstalování balíčků potřebných pro spuštění zdrojových kódů je možné použít příkaz

```
pip3 install -r requirements.txt
```

## Složka `prep`

Obsahuje skripty pro tvorbu datasetů ze surových dat ze SpokenData.

### `text_extract.py`
Příklad spuštění:

```bash
python3 text_extract.py sample_data/ -f
```

Extrahuje všechny manuální přepisy ze surových SpokenData dat ve složce `sample_data` pro účely adaptace rozpoznávače pro anotaci na SpokenData. Pro každý soubor přepisu ze SpokenData je vytvořen vlastní `.txt` soubor, který obsahuje všechny segmenty z přepisu na samostatném řádku. Tvar přepisu je ve formátu zvoleném na základě přepínače `-f` a `-s`.

Výstup je ve složce `[dir]_extracted_[f|s]` ve stejné rodičovské složce, jako je `[dir]` předaný skriptu při spuštění.

### `text_extract_with_IDs.py`

Příklad spuštění:

```bash
python3 text_extract_with_IDs.py sample_data/ -f
```

Tento skript extrahuje přepisy dle zvoleného formátu podobně jako skript předešlý. Výstup však ukládá do souborů `JSON` spolu s pseudo ID řídícího, pokud se alespoň v jednom vysílání (segmentu) daného souboru nachází. Díky tomu bude dále možné roztřídit nahrávky dle řečníků. 

Výstup je ve složce `[dir]_extracted_jsons_[f|s]`.
Struktura jednoho výstupního souboru může vypadat následovně:

```JSON
[
    {
        "sentence": "OKIUL24 Kunovice Information dobrý den\n",
        "ATCo_ID": "04"
    }
]
```

### `test_data_prepare.py`

Příklad spuštění:

```bash
python3 test_data_prepare.py sample_data/ -f -u
```

Skript slouží pro zpracování audia, které je určeno pro testovací množiny. Zpracovává surové přepisy ze SpokenData. Každou zpracovávanou nahrávku dle časových známek v přepisu v `.xml.trans` souboru rozstříhá na jednotlivá vysílání, zpracuje přepis, určí u nich jazyk a uloží je v podobném formátu jako skript předešlý.

Pracuje v několika režimech, které usnadňují tvorbu datasetů pro různé testovací množiny:

- `-u` přepínač slouží pro zpracování vysílání řídících z testovací množiny. Ze všech nahrávek, ve kterých se řídící vyskytují, extrahuje skript pouze vysílání těchto řídících. Z této zpracované množiny pak vznikne testovací množina neviděných řídících.
- `-o` přepínač také pracuje pouze s nahrávkami, ve kterých se vyskytuje řídící z testovací množiny, vystříhává z nahrávek však pouze ta vysílání, kde se řídící nevyskytuje. Z toho vznikne testovací množina neviděných vysílání letadel.
- `-a` přepínač slouží ke zpracování všech nahrávek a všech vysílání v dané složce. Tímto způsobem vzniká testovací množina z jiného letiště.

Výstupem je složka ve stejném adresáři jako složka předaná při spuštění skriptu. Ve výstupní složce se nachází pak složky `audio` a `transcripts_[full|short]`, kde jsou rozstríhané nahrávky a jejich přepisy. Pro tvorbu testovacích datasetů se pak skriptu popsanému níže předávají tyto dvě složky s daty.

### `dataset_prepare.py`

Příklad spuštění:

```
python3 dataset_prepare.py sample_data/ sample_data_extracted_jsons_f/ -t
```

```
python3 dataset_prepare.py audio/ transcripts/ -v
```

Skript načítá `JSON` soubory a provádí tvorbu datasetu ve dvou režimech. Přepínač `-t` je pro tvorbu trénovacího datasetu, přepínač `-v` pak pro testovací dataset. Rozdíl mezi těmito režimy je v tom, že trénovací režim do datasetu nezahrnuje přepisy, které mají v `JSON` souboru `ATCo_ID` řídících z testovací množiny. V režimu pro testovací množiny skript bere všechny soubory ze složky a přidává k nim navíc informaci o jazyku v nahrávce. Výstup pro např. testovací množinu má pak následující strukturu (u příkladu pro trénovací množinu by pouze nebyl přítomný jazyk):

```JSON
[
    {
        "audio": {
            "path": "sample_audio.wav",
            "array": [
                0.0,
                0.0,
                ...
            ],
            "sampling_rate": 16000
        },
        "sentence": "Oscar Kilo Alpha Bravo Charlie dobrý den\n",
        "language": "cs"
    },
    ...
]
```


## Složka `train`

Složka obsahuje pouze jeden skript, kterým je hlavní trénovací skript modelu Whisper. Jeho kostra byla převzata z https://huggingface.co/blog/fine-tune-whisper a byl upraven pro potřeby trénování.

Příklad spuštění:
```bash
python3 train.py --testset test_datasets/ --trainset train_datasets/ --feature_etc_files whisper_model/ --model whisper_model/ --logging_dir logging/ --training_output_dir output/ --new_model_path trained_model/ --lr 3e-5 --warmup 0.12 --batch_size 2 --num_epochs 45
```

Trénovací skript načte všechny datasety z předaných složek a udělá z každé složky jeden dataset.

## Složka `eval`

Tato složka slouží ke zjišťování statistik o datech. Obsahuje 4 skripty:

### `wer_ev.py`

Příklad spuštění:

```bash
python3 wer_ev.py --processor whisper_model/ --model whisper_model/ --unseen_ATC unseen_ATC/ --unseen_airport unseen_airport/ --unseen_LKKU unseen_LKKU/
```

### `stats.py`

Příklad spuštění:

```bash
python3 stats.py sample_data/ -f -a
```

Skript shromažďuje statistické informace o nahrávkách ze surových dat ze SpokenData. Je možné zpracovávané nahrávky filtrovat stejnými přepínači jako ve skriptu `test_dataset_prepare.py`, navíc je zde ale přepínač `-i`, který slouží pro zahrnutí pouze nahrávek, ve kterých se nevyskytují v žádném vysílání řídící z testovacích množin.

Skript následně na výstupu tiskne tabulku, která poskytuje uživateli informace o počtu zpracovaných nahrávek, počtu vysílání, počet vysílání dle jazyků, počet slov (ten záleží na zvoleném módu), celkové délce vysílání i počet různých "řečníků". Slovo řečníci je zde však ne příliš vhodné, jelikož skript počítá tyto řečníky dle různých volacích znaků letadel. Je přítomna také jednoduchá kontrola pro výskyt plného a zkráceného volacího znaku, aby se nezapočítávaly dvakrát, ovšem je nutno brát tohle číslo s rezervou.

### `wer_ev_spokenData.py`

Skript pro spuštění evaluovacího skriptu WER pro surové přepisy ze SpokenData. Skript porovnává původní automatický přpeis rozpoznávače s manuálním přepisem. WER je vyhodnocov8no pouze na plných přepisech, jelikož v této formě rozpoznávač přepisoval.

Příklad použití:

```bash
python3 wer_ev_SpokenData.py --raw_SpokenData_dir sample_data/
```

### `call_sign_error_rate.py`

Skript prochází výstup z `wer_ev.py` obsahující referenční a skutečné přepisy a předkládá je uživateli. Ten následně zadává počet slov volacích znaků v jednotlivých nahrávkách, která byla nahrazena, vynechána, jsou v přepisu navíc, a která byla správně určena. Na konci skript spočítá celkový WER na volacích znacích.

Příklad použití:

```bash
python3 call_sign_error_rate.py --file wer_ev_output.out
```
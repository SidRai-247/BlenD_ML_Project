# Spain KB - Quality Report

## Result
150 evidence cards, exactly 25 per category, IDs SPAIN_<CATEGORY>_0001-0025 (all unique, sequential). All 150 lines parse as JSON with the exact schema fields; every record has region "Spain", a direct https source URL, 3-8 keywords, and a taxonomy topic valid for its category.

## Category distribution
| Category | Cards |
|---|---|
| food | 25 |
| sports | 25 |
| family | 25 |
| education | 25 |
| holidays_celebrations_leisure | 25 |
| work_life | 25 |

Distinct domains per category: {"food": 14, "sports": 14, "family": 8, "education": 21, "holidays_celebrations_leisure": 16, "work_life": 14} (minimum 8, so the >=2 rule per category is met).

## Source distribution
64 distinct domains; top: {"ine.es": 20, "lamoncloa.gob.es": 14, "ich.unesco.org": 10, "educacionfpydeportes.gob.es": 7, "turismoasturias.es": 6, "spain.info": 4, "comunidad.madrid": 4, "aragon.es": 4}. Largest share is ine.es at 20/150 = 13.3% (cap 20% met).

| Source type | Cards |
|---|---|
| government | 72 |
| reference | 35 |
| statistical_agency | 24 |
| UNESCO | 10 |
| local_cultural_institution | 4 |
| university | 3 |
| cultural_institution | 2 |

Source quality: {"high": 101, "medium": 49}. Medium cards (press, tourism portals, consumer/union bodies, student theses, school-site documents) are: SPAIN_FOOD_0006, SPAIN_FOOD_0014, SPAIN_FOOD_0015, SPAIN_FOOD_0017, SPAIN_FOOD_0024, SPAIN_FOOD_0025, SPAIN_SPORTS_0007, SPAIN_SPORTS_0008, SPAIN_SPORTS_0011, SPAIN_SPORTS_0012, SPAIN_SPORTS_0015, SPAIN_SPORTS_0016, SPAIN_SPORTS_0019, SPAIN_SPORTS_0020, SPAIN_SPORTS_0021, SPAIN_SPORTS_0022, SPAIN_SPORTS_0024, SPAIN_SPORTS_0025, SPAIN_FAMILY_0016, SPAIN_FAMILY_0017, SPAIN_EDUCATION_0003, SPAIN_EDUCATION_0005, SPAIN_EDUCATION_0006, SPAIN_EDUCATION_0007, SPAIN_EDUCATION_0008, SPAIN_EDUCATION_0013, SPAIN_EDUCATION_0014, SPAIN_EDUCATION_0015, SPAIN_EDUCATION_0016, SPAIN_EDUCATION_0018, SPAIN_EDUCATION_0022, SPAIN_EDUCATION_0023, SPAIN_EDUCATION_0024, SPAIN_EDUCATION_0025, SPAIN_HOLIDAYS_0008, SPAIN_HOLIDAYS_0013, SPAIN_HOLIDAYS_0014, SPAIN_HOLIDAYS_0015, SPAIN_HOLIDAYS_0017, SPAIN_HOLIDAYS_0018, SPAIN_HOLIDAYS_0024, SPAIN_HOLIDAYS_0025, SPAIN_WORK_0007, SPAIN_WORK_0015, SPAIN_WORK_0016, SPAIN_WORK_0017, SPAIN_WORK_0018, SPAIN_WORK_0024, SPAIN_WORK_0025.

## Regional distribution (subregion mentions; multi-region cards counted once per region)
| Subregion | Cards |
|---|---|
| national/unspecified | 56 |
| País Vasco | 16 |
| Galicia | 12 |
| Comunidad Valenciana | 12 |
| Cataluña | 11 |
| Asturias | 8 |
| Andalucía | 8 |
| Comunidad de Madrid | 8 |
| Aragón | 8 |
| Canarias | 8 |
| Castilla y León | 5 |
| Cantabria | 4 |
| Navarra | 3 |
| Extremadura | 2 |
| Illes Balears | 2 |
| Castilla-La Mancha | 2 |
| Región de Murcia | 2 |
| Ceuta | 1 |
| La Rioja | 1 |

56 cards are national. No autonomous community exceeds 16 cards (País Vasco, 10.7%). Regions with 0-2 cards: Extremadura, Illes Balears, Castilla-La Mancha, Región de Murcia, La Rioja, Ceuta; Melilla has none.

## Language, scope, time
Original language: {"es": 134, "ca": 5, "en": 10, "gl": 1}. Cultural scope: {"national": 56, "regional": 64, "community": 10, "local": 20}. Time scope: {"contemporary": 108, "historical_to_contemporary": 39, "historical": 3}.

## Strongest sources
INE press notes and the Ministry of Education/Sports surveys and statistics, UNESCO Intangible Heritage entries, Turismo de Asturias, Junta/Xunta pages (Galicia, Extremadura, Andalusia school documents), the Basque Government pages and Eustat.

## Limitations (please read)
- **Verification depth:** each card was written from the page text returned in search results for the cited URL; only two La Moncloa pages were fetched in full. Figures in PDFs come from excerpts. A full manual re-check of each URL is still advisable before publication.
- **Titles:** some source.title values were taken from search-result headings or, where no headline was shown, from the page/URL slug (for example Deia, Ara, Turespaña 'Fin de semana en Badajoz', the Madrid press release, Euskaltel blog pages). Treat these as descriptive.
- **Medium-quality sources:** 49 of 150 cards are medium; press and tourism pages are secondary to institutions.
- **Data vintage:** several statistics are older (ECH 2018-2020, Eustat 2016-17 and 2020-21, 2004 Galician shellfishing plan, 2017 Rioja harvest note); time_scope reflects this only coarsely.
- **Version conflicts:** INE's conciliation module shows figures that differ between the PDF (used for the 38.3/42.1% and grandchild-care cards) and the updated web page (used for the 33.4% card).
- **Multinational UNESCO elements:** Mediterranean diet and transhumance are shared with other countries; Spain-specific scope is limited to the Spanish communities named in the source (Soria; unspecified for transhumance).
- **Coverage gaps:** no Basque-language cards and one Galician-language card; no family_gatherings_and_traditions cards; one school_traditions card; Melilla absent; few cards for Extremadura, Baleares, Murcia, La Rioja.
- **Near-duplicates:** several cards share a source page but state different facts (checked by text-prefix and URL/topic review, not by semantic dedup).
- **Hosting:** the ECH 2020 INE publication is cited from a University of Valencia mirror (uv.es).
- **Paraphrase:** English-language UNESCO texts were paraphrased in text_original; Spanish text_original fields are paraphrases in Spanish, not verbatim quotes.

## Contamination
BLEnD questions, answers, annotations or KB were not used; no card or source refers to BLEnD. The US KB was used only for schema and style.

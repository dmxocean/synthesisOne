# Translation Tasks Optimizer

## Project Overview

The Translation Tasks Optimizer is a system designed to solve the complex challenge of assigning translation tasks to translators. 

In the translation industry, project managers face the daily challenge of matching numerous translation tasks to qualified translators while balancing multiple competing factors including quality requirements, deadlines, translator expertise, language pair specialization, and cost constraints.

This project offers a data-driven solution that automates and optimizes the task assignment process. By analyzing historical translation data, translator profiles, client requirements, and scheduling constraints, the system creates optimal matches that maximize overall quality and efficiency while minimizing costs.

The core of the system leverages Constraint Satisfaction, Machine Learning, and Deep Learning approaches to identify optimal translator candidates for each task. These complementary technologies analyze historical performance data, language pair proficiency, and completion patterns to detect the most suitable translators while respecting schedule constraints and client requirements. 

This approach not only improves operational efficiency but also enhances translator satisfaction through fairer workload distribution and client satisfaction through better quality outcomes.

## Project Structure

```
❯ tree
.

TODO

```

## Datasets and Data Analysis

The project uses four primary datasets that provide comprehensive information about the translation process ecosystem. These datasets were thoroughly analyzed to extract insights and patterns that inform the optimization model.

### Client Dataset (`clients.csv`)
Contains information about client profiles, requirements, and preferences.
| Column | Type | Description | Analysis Insights |
| :--------------------- | :---- | :--------------------------------------------- | :---------------------------------------------------- |
| `CLIENT_NAME` | String | Unique identifier for each client | 2,567 unique clients identified |
| `SELLING_HOURLY_PRICE` | Float | Hourly rate charged to the client | Range: 20.00€-90.00€, Average: 26.17€ |
| `MIN_QUALITY` | Float | Minimum quality threshold required (1-10 scale) | Average requirement: 5.57/10 |
| `WILDCARD` | String | Requirement that can be bypassed if necessary | "Deadline" is most common (38.06%), followed by "Price" (36.46%) |

**Analysis Summary:** 
- **Client Segmentation** (K-Means clustering):
  * **Cluster 0: Standard Quality Focus (66.2%)**
    - Price: Moderate (avg 24.1€)
    - Quality: High (avg 7.35/10)
    - Wildcard: Typically "Deadline"
  * **Cluster 1: Price Sensitive (22.5%)**
    - Price: Moderate (avg 24.68€)
    - Quality: Minimal (avg 0.0/10)
    - Wildcard: Typically "Price"
  * **Cluster 2: Premium (11.3%)**
    - Price: High (avg 41.35€)
    - Quality: Moderate (avg 6.2/10)
    - Wildcard: Typically "Deadline"
- **Key Insight**: Higher pricing ≠ higher quality demands. Premium clients often prioritize other factors over top-tier quality scores.

### Translator Schedules Dataset (`schedules.csv`)
Contains detailed information about translator availability and working patterns.
| Column | Type | Description | Analysis Insights |
| :-------------- | :------ | :----------------------------------------- | :-------------------------------------------------------- |
| `NAME` | String | Translator identifier | 871 unique translators |
| `START` | String | Time when translator begins their workday | Most common: 09:00 (10h shifts are common) |
| `END` | String | Time when translator ends their workday | Most common: 19:00 |
| `MON` through `SUN` | Integer | Day availability (1=available, 0=unavailable) | 57.18% work weekends (`SAT`: 46.96%, `SUN`: 47.53%) |
| `WEEKLY_HOURS` | Float | Calculated total weekly available hours | Average: 48.97 hours, Median: 50 hours |
| `LIKELY_TIMEZONE` | String | Estimated timezone based on start hour | Americas (94.49%), Asia/Pacific (4.02%), Europe/Africa (1.49%) |

**Analysis Summary:** 
- **Availability Patterns**:
  * Peak availability: Friday at 13:00 (69.46% translators)
  * Weekend coverage: 57.18% available
  * Geographical distribution: Americas dominance (94.49%)
- **Work Schedule Types**:
  * Full-time (≥5 days): 64.87%
  * Flexible schedules: 29.51%
  * Occasional workers: 20.90%
- **Operational Metrics**:
  * Average utilization: 29.15%
  * Unusual schedules: 51.78% (high hours, single-day availability, overnight shifts)
- **Key Challenge**: Limited Europe/Asia coverage creates 24/7 service gaps

### Translation Task History Dataset (`data.csv`)
Contains historical data on translation tasks, including quality evaluations, costs, client details, and timestamps.

| Column| Type | Description | Analysis Insights |
| :-------------------------- | :------- | :-------------------------------------------------------------------------- | :--------------------------------------------------------- |
| `PROJECT_ID`| Integer | Unique project identifier | 13,059 unique projects |
| `TASK_ID`| Integer | Task identifier within project | 554,024 unique tasks (out of 554,029 total rows) |
| `PM` | String| Project manager assigned | 4 unique PMs with varying workloads |
| `TASK_TYPE` | String| Category of translation service (See initial docs for details) | "Translation" is most common (50.21%) |
| `START`| Datetime | Task start date/time| Used for planning projections |
| `END` | Datetime | Theoretical task delivery date (can be used to compare with the DELIVERED date to check for delays) | 90.43% of tasks completed on time |
| `ASSIGNED` | Datetime | Timestamp when task is pre-assigned to translator | Part of workflow timing analysis |
| `READY` | Datetime | Timestamp when translator notified they can start | '`READY`-to-`WORKING`' stage avg: 17.6h (potential bottleneck) |
| `WORKING`| Datetime | Timestamp when translator starts the task| Part of workflow timing analysis |
| `DELIVERED` | Datetime | Timestamp when translator delivers the task | Compared with `END` for delays |
| `RECEIVED` | Datetime | Timestamp when PM receives the delivered task | Part of workflow timing analysis |
| `CLOSE` | Datetime | Timestamp when PM marks task as completed | End of task lifecycle |
| `SOURCE_LANG` | String| Original content language| 34 distinct source languages |
| `TARGET_LANG` | String| Translation target language | 68 distinct target languages |
| `LANGUAGE_PAIR` | String| Combined Source > Target language (Derived) | English > Spanish (Iberian) dominates (50.73% of tasks) |
| `TRANSLATOR`| String| Assigned translator | Average: ~626 tasks per translator |
| `QUALITY_EVALUATION` | Integer | Quality score (e.g., 1-10) assigned after review | Average score: 7.06/10 |
| `FORECAST` | Float | Estimated hours required for the task | Used for planning and cost estimation |
| `HOURLY_RATE` | Float | Historical hourly rate paid to the translator for this specific task | Varies by translator, language pair, task type |
| `COST` | Float | Total cost paid to the translator for the task (`HOURLY_RATE` * `FORECAST`) | Average: 38.26€ per task|
| `MANUFACTURER` | String| Client organization name | Corresponds to `CLIENT_NAME` in `clients.csv` |
| `MANUFACTURER_SECTOR` | String| Level 1 client industry sector | 'Information Technology' most common (34.31%) |
| `MANUFACTURER_INDUSTRY_GROUP` | String| Level 2 client industry categorization | Provides finer client segmentation |
| `MANUFACTURER_INDUSTRY` | String| Level 3 client industry categorization | Provides finer client segmentation |
| `MANUFACTURER_SUBINDUSTRY` | String| Level 4 client industry categorization | Provides most granular client segmentation |

**Analysis Summary:**
- **Task Composition**:
  * Dominant language pair: English > Spanish (Iberian) - 50.73%
  * Top industry sector: Information Technology - 34.31%
  * Most common task type: Translation - 50.21%
- **Performance Metrics**:
  * On-time completion: 90.43%
  * Average quality score: 7.06/10
  * Average task cost: 38.26€
- **Workflow Insights**:
  * Bottleneck stage: Ready → Working (17.6h average)
  * Cost-quality correlation: Very weak (-0.0149)
- **Key Finding**: Pricing appears driven by language pair, task type, and urgency rather than by expected quality outcomes.

### Translator Cost Pairs Dataset (`translatorsCostPairs.csv`)
Maps translators to language pairs with associated rates.
| Column | Type | Description | Analysis Insights |
| :---------- | :------ | :-------------------------------- | :-------------------------------------------------------- |
| `TRANSLATOR` | String | Translator identifier | 871 unique translators |
| `SOURCE_LANG` | String | Source language | 34 unique source languages |
| `TARGET_LANG` | String | Target language | 68 unique target languages |
| `HOURLY_RATE` | Integer | Historical rate paid to the translator for the specific task | Range: 8.00€-60.00€, Average: 20.61€ |
| `LANGUAGE_PAIR` | String | Calculated Source > Target pair | 258 unique pairs |

**Analysis Summary:** 
- **Language Pair Coverage**:
  * Total unique pairs: 258
  * Single-translator pairs: 75 (29.07%)
  * Most common: English > Spanish (Iberian)
- **Translator Specialization**:
  * Single pair specialists: 32.26%
  * Highly versatile (>10 pairs): 8.50%
  * Specialization-rate correlation: -0.12
- **Rate Economics**:
  * Rare language premium: 79% higher than common languages
  * Average rates: Common (18.32€) vs. Rare (32.76€)
- **Market Segmentation** (K-Means clustering):
  * **Budget Common**: Low rate, high translator count
  * **Premium Specialized**: Moderate rate, low translator count
  * **Standard Common**: High rate, low translator count
  * **Rare Specialty**: Moderate rate, moderate translator count

## Installation Process

To set up the Translation Tasks Optimizer on your system, follow these steps:

### Prerequisites
* [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
* Git

### Setup Steps
```bash
# Clone the Repository
git clone https://github.com/dmxocean/synthesisOne
cd synthesisOne

# Create Conda Environment
conda create -n translationEnv python=3.10 -y
# Activate Conda Environment
conda activate translationEnv

# Install Dependencies
pip install -r requirements.txt
```

Upon completion of these steps, the system should be ready for use within the `translationEnv` conda environment. Any issues encountered during installation can be reported through the project's issue tracker.
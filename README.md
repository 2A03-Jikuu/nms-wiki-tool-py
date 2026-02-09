# 🚀 No Man's Sky Wiki Page Generators (Python Version)

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Compatible-yellow?logo=googlecolab&logoColor=white)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
![Status](https://img.shields.io/badge/Status-Proof%20of%20Concept-red)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)
![No Man's Sky](https://img.shields.io/badge/No%20Man's%20Sky-Community%20Tool-purple)
![AI Assisted](https://img.shields.io/badge/AI-Assisted%20Development-blueviolet)

**A collection of interactive, form-based tools that help players of No Man's Sky create properly formatted wiki pages for their in-game discoveries, no wiki markup knowledge required.**

> ⚠️ **This is a proof of concept.** These tools are functional demonstrations of how game-related procedural logic can be studied and packaged into user-friendly community tools. They are not polished, production-ready applications. Expect rough edges, and feel free to build on top of what is here.

> 📋 **Intellectual Property Notice:** No Man's Sky and all related game data, assets, names, and mechanics are the intellectual property of **Hello Games Ltd.** This project is a fan-made community tool created for non-commercial purposes. All game-related data referenced or used by these tools belongs to Hello Games.

> 🤖 **AI Disclosure:** This project was developed with the assistance of **AI language models (LLMs)** during the coding process. AI tools were used to help with code generation, debugging, problem-solving, and documentation writing throughout development. While all output has been reviewed, tested, and adjusted by a human, transparency about the use of AI assistance is important. The overall project direction, design decisions, testing, and domain-specific knowledge (game mechanics, wiki conventions, community standards) are human-driven.

---

## What Is This?

If you have ever tried to contribute a page to the [No Man's Sky Wiki](https://nomanssky.fandom.com/), you know the pain: dozens of template fields, strict formatting conventions, portal glyph conversions, coordinate math, and procedurally generated region names that you cannot just guess. Doing it by hand is tedious, time-consuming, and easy to get wrong.

**This project explores whether all of that can be automated.**

Each tool in this collection is a guided, tab-by-tab form that runs inside a **Jupyter Notebook** or **Google Colab**. You fill in what you know about your discovery, and the tool handles everything else: coordinate conversion, procedural name generation, AGT Stardate calculation, data validation, and final template rendering. The output is a block of wiki markup text you can preview, copy to your clipboard, or download as a file, ready to paste directly into a wiki editor.

The main technical challenge this project tackles is **faithfully reproducing the game's procedural region name generation algorithm in Python**, including C++-style byte-level arithmetic, overflow behavior, weighted character transition tables, and linguistic post-processing rules. The result matches exactly what you would see in-game, which means contributors never have to manually look up region names again.

---

## 💡 Inspiration

The initial idea for this project was inspired by **[Lenni009](https://github.com/Eisvana)** and the **[Eisvana Wiki Page Creator](https://github.com/Eisvana/EisvanaWikiPageCreator)**, a web-based wiki page tool that many No Man's Sky communities have adapted for their own use. That project demonstrated how much easier wiki contributions become when you wrap the formatting logic in an interactive interface.

This project takes a **different approach** out of practical necessity. Rather than building a web application, which would require front-end development skills I have not yet built up to a comfortable level, I chose to work within **Jupyter Notebooks and Google Colab**, an environment I am more familiar with and one that still allows for rich interactive forms through `ipywidgets`. The trade-off is that these tools are less portable than a web app, but they are functional, accessible to anyone with a browser (via Colab), and were a realistic scope for my current skill set.

---

## 📌 About This Project & AGT

This is an **unofficial** wiki tool project built around the workflows and data conventions of the [Alliance of Galactic Travellers (AGT)](https://nomanssky.fandom.com/wiki/Alliance_of_Galactic_Travellers), one of the major community civilizations in No Man's Sky.

The AGT maintains their own **in-house wiki tools** that their members use internally for documentation. Much of the foundational data, template structures, field conventions, and formatting logic used in this project were **adapted from those official AGT tools**. This project essentially takes that existing knowledge base and repackages it into a standalone, notebook-based format that anyone can run, serving as both a learning exercise and a proof of concept for how community wiki tooling could work in an open, accessible way.

**To be clear:** this project is not officially affiliated with or endorsed by the AGT. It is an independent effort that builds on publicly available information and community conventions. If you are an AGT member looking for the official tooling, please refer to your community's internal resources.

---

## ⚠️ Known Limitations & Expectations

I want to be upfront about the state of this project. While I have tested each tool down to the smallest details I could think of and have done my best to catch edge cases, **there are likely still caveats and bugs that I have not yet uncovered**. I am still growing as a developer and currently sit at what I would describe as an intermediate level when it comes to Python and debugging. That means some issues may only surface when other people use these tools in ways I did not anticipate or with data combinations I did not think to test.

**What you should expect:**
- The tools work correctly for the scenarios I have been able to test thoroughly
- There may be edge cases, unexpected input combinations, or environment-specific quirks that produce errors or incorrect output
- Some parts of the code could definitely be written more cleanly or efficiently
- Results may vary depending on your specific setup, notebook environment, or the data you enter

If you do run into a problem, I genuinely appreciate bug reports. Every issue someone finds helps me learn and makes the tools better for everyone. Please do not hesitate to open an issue, even if you are not sure whether something is actually a bug or just something you are unsure about. I would rather hear about it than not.

---

## 🛠️ Available Tools

| Tool | Description |
|------|-------------|
| 🪐 **Planet / Moon** | Biome classification, environmental sensor readings, geological data, resource lists, points of interest, and gallery images. |
| 🌟 **Star System** | Faction info, star characteristics, demographics, space station details, and all calculated location fields. |
| 🗺️ **Region** | Enter portal glyphs and get a complete region wiki page with procedural name, coordinates, distance from center, and quadrant. |
| 🚀 **Starship** | Dynamic dropdowns for parts based on ship type, auto-calculated inventory sizes, spawn chances by economy level, and full color/stats breakdowns. |
| 🔫 **Multi-Tool** | Type, class, stats, colors, location data, and automatically generated region names from portal glyphs. |
| 🏠 **Base** | Builder credits, construction dates with auto-converted AGT Stardate, facilities, nearby points of interest, and gallery support. |
| 🐾 **Fauna** | Genus, diet, temperament, gender details, measurements, and behavioral notes. |
| 🌿 **Flora** | Biological traits, nutrient sources, harvestable resources, and age data. |
| 🪨 **Mineral** | Geological properties, formation details, resource yields, and metal content. |

---

## ✨ What It Demonstrates

### 🔢 Automatic Coordinate Conversion
The game uses a system of **portal glyphs**, 12-character hexadecimal codes that encode a location in 3D galactic space. Manually converting these into usable coordinates requires understanding the game's internal coordinate system and doing hexadecimal math. These tools handle that automatically.

When you enter a portal glyph code and select a galaxy, the tool instantly calculates:
- **Galactic coordinates** (X, Y, Z axes and system index) from the raw glyph data
- **Distance from the galactic center** in light-years
- **Galactic quadrant designation** (Alpha, Beta, Gamma, or Delta) based on the X and Z coordinate signs

All of this updates in real time as you type, so you can see the results immediately without clicking any buttons.

### 🧬 Procedural Region Name Generation
This is the centerpiece of the project from a technical standpoint.

Every region in No Man's Sky has a name that is not stored in a database anywhere. Instead, the game **generates it on the fly** from the region's coordinates using an algorithm involving seed values, pseudo-random number sequences, weighted character probability tables, and a set of linguistic rules that clean up the output to look like a plausible sci-fi name.

These tools **reproduce that same algorithm in Python**. The implementation handles byte-level seed manipulation, character transition probabilities, vowel/consonant balancing, and various edge cases in the naming rules. The generated names match what you would see in-game for any given set of coordinates, which means contributors never have to manually look up or screenshot region names again.

The Python implementation was adapted from existing community work in C# (see Acknowledgments), with significant rework needed to account for differences in how Python and C++ handle integer overflow, unsigned byte arithmetic, and type casting.

### 📅 AGT Stardate Calculation
The AGT community uses a custom date format called the **AGT Stardate** for recording discovery dates on wiki pages. Rather than requiring users to calculate this format manually, the tools automatically convert standard real-world dates into the correct AGT Stardate representation.

### 📋 Dynamic Game Data
Instead of hardcoding dropdown options that would go stale after every game update, the tools **fetch reference data from community-maintained online repositories** at runtime. This includes:
- Galaxy names and indices
- Biome and weather types
- Elements and resources
- Ship parts, wing styles, and thruster types
- Creature genera and temperaments
- And many other game-specific classification lists

This means the tools stay reasonably up to date without needing code changes every time the game receives a patch.

### ✅ Input Validation & Output Generation
Before generating the final wiki markup, each tool **validates user inputs** to catch common mistakes: missing required fields, invalid glyph codes, out-of-range values, and formatting issues. Error messages are displayed inline so users know exactly what needs to be fixed.

Once validation passes, the tool assembles a complete wiki page using the proper template format, including:
- Infobox templates with all required parameters
- Formatted section headers and body text
- Resource and point-of-interest tables
- Gallery sections for images and videos
- Navigation categories and tags

The generated output can be:
- **Previewed** in a scrollable output area within the notebook
- **Copied** directly to your clipboard with one click
- **Downloaded** as a `.txt` file for later use

### 📝 Example Data
Every tool includes a **"Load Example"** button that fills the entire form with sample data so you can immediately see what a completed output looks like without entering anything yourself. Examples are provided for typical use cases (for instance, the Planet tool includes separate example data for both a planet and a moon) to help new users understand what kind of information goes where.

---

## 🚀 Getting Started

### Google Colab (Easiest Way to Try It)

1. Open the notebook for the tool you want to use
2. Click **Runtime → Run all**
3. A tabbed form will appear, fill it out
4. Click **Generate** to produce your wiki markup
5. Copy or download the output

### Running Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open any notebook and run all cells.

---

## 📦 Requirements

- Python 3.7+
- Jupyter Notebook or Google Colab
- `ipywidgets` for the interactive form interface
- Internet connection (for fetching live game data from community repositories)

---

## 🎯 Who Might Find This Useful?

- **Wiki contributors** who want to submit properly formatted pages without wrestling with markup syntax
- **No Man's Sky players** curious about documenting their discoveries
- **Developers** interested in how the game's procedural name generation works and how it can be reproduced in Python
- **Other NMS communities** looking for a starting point to build their own wiki tooling

---

## 📚 Helpful Resources

If you are interested in understanding the procedural region name generation code and want to study the underlying concepts further, the following book is a closely related resource:

> **"Hacker's Delight" by Henry S. Warren Jr.**
>
> This book covers the kind of low-level bit manipulation, byte-level arithmetic, and overflow behavior techniques that are central to how the region name generation algorithm works. Much of the work involved in adapting the original C# and C++ logic into Python required understanding these concepts, and this book is one of the best references available on the subject.

---

## 🤝 Contributing & Forking

Since this is a proof of concept, there is plenty of room for improvement: cleaner code, better error handling, additional tools, or even porting the whole thing to a standalone web app. If any of that interests you, feel free to open an issue or submit a pull request.

Even if you are not a developer, **bug reports and feedback are incredibly valuable**. If something does not work the way you expected, or if the generated output does not look right, please let me know. Every report helps me improve.

### 🍴 Forking This Project

If you have a better idea for how these tools should work, a different technical approach you want to try, or you want to adapt this for your own NMS community, **you are absolutely welcome to fork this repository and make it your own**. That is part of why this project exists as an open proof of concept in the first place.

A few things to keep in mind if you fork:

- ✅ You are free to modify, adapt, and redistribute the code however you see fit
- ✅ You are welcome to take the project in a completely different direction if that works better for your needs
- 📋 Please give appropriate credit and link back to this repository and the original contributors listed in the Acknowledgments section
- 🚫 **The CC BY-NC 4.0 license still applies to your fork.** This means you may not use the forked code, or any adaptation of it, for commercial purposes. This restriction carries forward to all derivative works. Please respect this, as much of the foundational work here was shared freely by community members for the benefit of the NMS community, not for commercial gain

If you do build something cool from this, I would love to hear about it. Feel free to drop a link in the issues or discussions.

---

## 📄 License

This project is licensed under the **[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)**.

### What this means:

- ✅ **You are free to** share, copy, redistribute, adapt, remix, and build upon this work
- ✅ **You must** give appropriate credit, provide a link to the license, and indicate if changes were made
- 🚫 **You may not** use this work, or any adaptation of it, for **commercial purposes**

This license applies to **all code, templates, configurations, and adaptations** contained in this repository. The non-commercial restriction is intentional. Much of the foundational work in this project draws from open-source community contributions and collaborative efforts by people who shared their work freely for the benefit of the No Man's Sky community. Applying a non-commercial license ensures that this spirit is preserved and that neither this project nor any derivative of it can be monetized or packaged into a commercial product.

**Game data and assets** referenced by these tools remain the intellectual property of **Hello Games Ltd.** and are used here under fair use for fan-made, non-commercial community purposes.

---

## 🙏 Acknowledgments

- **[Lenni009](https://github.com/Eisvana)** and the **[Eisvana Wiki Page Creator](https://github.com/Eisvana/EisvanaWikiPageCreator)** for the original inspiration. The idea of wrapping wiki formatting logic in an interactive tool that anyone can use came directly from seeing what the Eisvana community built, and many NMS communities have since adapted that approach.

- **[celabgalactic](https://github.com/celabgalactic) & the SLT team** for the AGT database and wiki data that served as the foundation for this project. The template structures, field conventions, and game data used across these tools were adapted from their work on the AGT's in-house wiki tooling.

- **[andraemon](https://github.com/andraemon/SystemNameCalculator)** for the original region name lookup implementation in C#. The procedural region name generation code in this project was directly inspired by his SystemNameCalculator. His C# implementation was studied, reworked, and adapted into Python with significant adjustments to handle language differences, byte-level arithmetic behavior, and overflow semantics between the two languages.

- **[monkeyman192](https://github.com/monkeyman192/NMS_translate)** for the `letter_map.json` character transition probability data used in the name generation process. The work of extracting these tables from the game's own data made it possible for andraemon's C# tool to exist in the first place, and by extension made this Python adaptation possible as well.

- The **[Alliance of Galactic Travellers (AGT)](https://nomanssky.fandom.com/wiki/Alliance_of_Galactic_Travellers)** community for their documentation conventions and standards

- The **[No Man's Sky Wiki](https://nomanssky.fandom.com/)** community for the templates and formatting standards these tools target

- **Hello Games** for creating No Man's Sky and the universe that inspires all of this community work

---

*Built with ❤️ for the No Man's Sky community*

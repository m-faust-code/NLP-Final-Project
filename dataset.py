import string
from types import NoneType
from typing import List
import wikipediaapi as wiki
import numpy as np 
import random

wiki_wiki = wiki.Wikipedia("wikipedia_network (mfaust@macalester.edu)", "en", extract_format=wiki.ExtractFormat.WIKI)

categories = ["Category:The arts", "Category:Health", "Category:History",
              "Category:Science", "Category:Religion", "Category:Technology", ]

def isCategoryOnlyCategory(category: wiki.WikipediaPage):
    result = True
    pages = category.categorymembers.values()
    for page in pages:
        if page.namespace != wiki.Namespace.CATEGORY:
            result = False
    return result


def samplePages(category: wiki.WikipediaPage, n, max_depth, otherCategories=None):
    pages = category.categorymembers
    if otherCategories is None:
        otherCategories = categories.copy()
        if category.title in otherCategories:
            otherCategories.remove(category.title)
    pageSample = random.choices(list(pages.values()), k=n)
    results = []
    reps = 0
    for page in pageSample:
        super_cats = [x.title for x in getSupercategories(page, 2)]
        if page.namespace == wiki.Namespace.CATEGORY and len(set(otherCategories) & set(super_cats)) == 0 and "Category:Wikipedia images by subject" not in super_cats:
            if max_depth > 1 or max_depth == 1 and not isCategoryOnlyCategory(page):
                results.extend(samplePages(page, 1, max_depth - 1, otherCategories))
            else:
                pageSample.append(random.choice(list(pages.values())))
                reps += 1
                if reps >= 100:
                    print(category)
                    return
        elif page.namespace == wiki.Namespace.MAIN and len(set(otherCategories) & set(super_cats)) == 0 and "Category:Wikipedia images by subject" not in super_cats:
            results.append(page)
        else:
            pageSample.append(random.choice(list(pages.values())))
            reps += 1
            if reps >= 100:
                print(category)
                return
    return results

def getSupercategories(page, max_depth) -> List[wiki.WikipediaPage]:
    pages = list(page.categories.values())
    pages = list(filter(lambda x: "Category:Hidden categories" not in x.categories.keys(), pages))
    results = pages.copy()
    if max_depth > 0:
        for category in pages:
            supers = getSupercategories(category, max_depth - 1)
            ids = [x.pageid for x in results]
            results.extend(x for x in supers if x.pageid not in ids)
    return results

def GetSummaryDataset(category: wiki.WikipediaPage, n):
    pages = samplePages(category, n, 4)
    text = []
    for page in pages:
        new_text = page.summary.split()
        text += new_text
        if len(new_text) == 0:
            print(page)
    return text

def makeFiles(n):
    for name in categories:
        if name not in ["Category:The arts", "Category:Health", "Category:History"]:
            category = wiki_wiki.page(name)
            filename = name.lower().removeprefix("category:").replace(" ", "") + str(n) + ".txt"
            print(filename)
            words = GetSummaryDataset(category, n)
            delimiter = ";"
            text = delimiter.join(words)
            f = open(filename, "w", encoding="utf-8")
            f.write(text)
            f.close()

def makeDevset(n):
    for name in categories:
        category = wiki_wiki.page(name)
        filename = name.lower().removeprefix("category:").replace(" ", "") + str(n) + "dev.txt"
        print(filename)
        texts = []
        for i in range(n):
            words = GetSummaryDataset(category, 1)
            delimiter = ";"
            text = delimiter.join(words)
            texts.append(text)
        delimiter = "\n"
        file_texts = delimiter.join(texts)
        f = open(filename, "w", encoding = "utf-8")
        f.write(file_texts)
        f.close

def cleanData(filename: str):
    f = open(filename, "r", encoding = "utf-8")
    text = f.read()
    words = text.split(";")
    clean_words = []
    for word in words:
        if len(word) == 0:
            clean_words.append(";")
        else:
            clean_words.extend(cleanWord(word.lower()))
    delimiter = " "
    clean_text = delimiter.join(clean_words)
    f.close()
    f = open("clean_" + filename, "w", encoding="utf-8")
    f.write(clean_text)
    f.close()

def cleanWord(word: str) -> List[str]:
    result = []
    if len(word) == 0:
        result.append("")
    elif word[0] in string.punctuation and word[-1] in string.punctuation:
        result.append(word[0])
        result.extend(cleanWord(word[1:-1]))
        result.append(word[-1])
    elif word[0] in string.punctuation:
        result.append(word[0])
        result.extend(cleanWord(word[1:]))
    elif word[-1] in string.punctuation:
        result.extend(cleanWord(word[:-1]))
        result.append(word[-1])
    else:
        result.append(word)
    return result
        

cleanData("thearts20dev.txt")

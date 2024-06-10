import requests
import json
from bs4 import BeautifulSoup
import csv

cant = 100


def build_url(problem):
    return f'https://codeforces.com/contest/{problem["contestId"]}/problem/{problem["index"]}'


def get_problem_text(problem_url):
    response = requests.get(problem_url)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    problem_statement_div = soup.find('div', {'class': 'problem-statement'})

    return problem_statement_div.contents[1].getText()


def get_problems():
    response = requests.get('https://codeforces.com/api/problemset.problems')

    if response.status_code != 200:
        return []

    return json.loads(response.text)['result']['problems'][:cant]


def write_dicts_to_csv(data, filename):
    if not data:
        return

    headers = ['contestId', 'index', 'name', 'rating', 'tags', 'text']

    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        for row in data:
            new_row = {}

            for k in headers:
                if k in row:
                    new_row[k] = row[k]

            if len(new_row) == len(headers):
                continue

            writer.writerow(new_row)


problems = get_problems()

for p in problems:
    url = build_url(p)
    p["text"] = get_problem_text(url)

write_dicts_to_csv(problems, 'data/data.csv')

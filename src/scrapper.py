import requests
import json
from bs4 import BeautifulSoup
import sys

cant = 100 if len(sys.argv) < 2 else int(sys.argv[1])
cant_sol = 10 if len(sys.argv) < 3 else int(sys.argv[2])


def build_url_text(problem):
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


def write_dicts_to_json(data, filename):
    headers = ['contestId', 'index', 'name',
               'rating', 'tags', 'text', 'solutions']

    filter_data = []

    for row in data:
        new_row = {}

        for k in headers:
            if k in row:
                new_row[k] = row[k]

        if len(new_row) != len(headers):
            continue

        filter_data.append(new_row)

    f = open(filename, mode='w')
    json.dump(filter_data, f)
    f.close()


def get_accepted_cpp_solutions(problem, count=100):
    base_url = f'https://codeforces.com/problemset/status/{problem["contestId"]}/problem/{problem["index"]}?page='
    solutions = []
    page = 1

    while len(solutions) < count:
        url = base_url + str(page)
        response = requests.get(url)
        if response.status_code != 200:
            break

        soup = BeautifulSoup(response.text, 'html.parser')
        rows = soup.find_all('tr')

        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 7:
                continue
            submission_id = row['data-submission-id']
            verdict = cells[5].get_text().strip()
            language = cells[4].get_text().strip()

            if verdict == 'Accepted' and 'C++' in language:
                submission_url = f'https://codeforces.com/contest/{problem["contestId"]}/submission/{submission_id}'
                submission_response = requests.get(submission_url)
                submission_soup = BeautifulSoup(
                    submission_response.text, 'html.parser')
                code_div = submission_soup.find(
                    'pre', {'id': 'program-source-text'})
                if code_div:
                    solutions.append(code_div.get_text())
                    if len(solutions) >= count:
                        break

        page += 1

    return solutions


problems = get_problems()

for p in problems:
    url = build_url_text(p)
    p["text"] = get_problem_text(url)
    p["solutions"] = get_accepted_cpp_solutions(p, cant_sol)

write_dicts_to_json(problems, 'data/data.json')

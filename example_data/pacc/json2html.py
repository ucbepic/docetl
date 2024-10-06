import json
import sys
import os
import csv

def filter_json(json_obj):
    if isinstance(json_obj, dict):
        return {k: filter_json(v) for k, v in json_obj.items() if k != "content"}
    elif isinstance(json_obj, list):
        return [filter_json(item) for item in json_obj]
    else:
        return json_obj

def json_to_html(json_obj):
    
    html_content = '<html><body><table border="1">'
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            if key not in ["content"]:
                html_content += f'<tr><th>{key}</th><td>{value}</td></tr>'

    elif isinstance(json_obj, list):
        for item in json_obj:
            html_content += '<tr>'
            if isinstance(item, dict):
                for key, value in item.items():
                    if key not in ["content"]:
                        html_content += f'<th>{key}</th><td>{value}</td>'
            else:
                html_content += f'<td>{item}</td>'
            html_content += '</tr>'

    html_content += '</table></body></html>'
    return html_content

def save_list_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in data:
            csvwriter.writerow(row)

def json_to_map(json_obj):
    if isinstance(json_obj, dict):
        return {k: json_to_map(v) for k, v in json_obj.items()}
    elif isinstance(json_obj, list):
        return [json_to_map(item) for item in json_obj]
    else:
        return json_obj
    

def get_councilmembers(jmap):
    councilmembers = set()
    for issue in jmap.get('issues', []):
        for v in issue.get('votes', []):
            councilmember = v.split(":")[0]
            councilmembers.add(councilmember)
    return sorted(councilmembers)

def json_to_list_table(json_obj):
    jmap = json_to_map(json_obj)

    table = []

    for meeting_day in jmap:
        councilmembers = get_councilmembers(meeting_day)

        #print(councilmembers)

        # Create the table header
        if not table:
            table_header = ["Date", "Category", "Issue", "Outcome"]
            table_header.extend(councilmembers)
            table_header.extend(["Rationale"])
            table.append(table_header)
    
        # Create the table rows
        table_rows = []

        for issue in meeting_day.get('issues', []):
            table_row = [meeting_day['date'], issue['category'], issue['issue'],issue['outcome']]
            table_row.extend(councilmembers)

            for v in issue.get('votes', []):
                councilmember = v.split(":")[0]
                vote = v.split(":")[1].strip()
                if councilmember in table_row:
                    index = table_row.index(councilmember)
                    table_row[index] = vote

            table_row.append(issue.get('rationale', ""))
            
            table.append(table_row)


    return table            
            

def list_to_html_table(data):
    html = '<table border="1">'
    for row in data:
        html += '<tr>'
        for cell in row:
            html += f'<td>{cell}</td>'
        html += '</tr>'
    html += '</table>'
    return html
    
def xjson_to_insight(json_obj):
    jmap = json_to_map(json_obj)

    councilmembers = {}
    issues = {}
    
    for item in jmap:
        citem = item.get('categories')

        category = citem.get("category")
        issue = citem.get("issue")
        votes = citem.get("votes")

        issue_votes = issues.get(issue, {})

        category_issues = categories.get(category, set())
        category_issues.add(issue)
        categories[category] = category_issues

        for v in votes:
            councilmember = v.split(":")[0]
            councilmember_vote = v.split(":")[1].strip()
            issue_votes[councilmember] = councilmember_vote

            councilmember_issues = councilmembers.get(councilmember, [])
            councilmember_issues.append({"issue": issue, "vote": councilmember_vote})
            councilmembers[councilmember] = councilmember_issues

        issues[issue] = issue_votes

    lastcat = ""

    htmlstr = "<table border=1px>"
    councilmember_keys = sorted(councilmembers.keys())
    htmlstr += "<tr><td>Issue</td>"
    for ckey in councilmember_keys:
        htmlstr += "<td>%s</td>" %ckey
    htmlstr += "</tr>"

    for category, categoryissues in categories.items():
        if lastcat != category:
            htmlstr += "<tr><td colspan='15'><h2>%s</h2></td></tr>" %category
            print("Category: %s" %category)

        for issuekey in categoryissues:
            issue = issues.get(issuekey, None)
            if not issue:
                continue
            print("  Issue: %s" %issuekey)
            htmlstr += "<tr>"
            htmlstr += "<td>%s</td>" %issuekey

            for councilmember in councilmember_keys:
                cvote = issue.get(councilmember, " - ")
                print ("    Councilmember: %s, Vote: %s" %(councilmember, cvote))
                htmlstr += "<td>%s</td>" %cvote


            htmlstr += "</tr>"

    htmlstr += "<table>"
    return htmlstr


    # print(issues.keys())
    # print(councilmembers.keys())


def main():
    if len(sys.argv) != 2:
        print("Usage: python json2html.py <json_filename>")
        sys.exit(1)

    json_filename = sys.argv[1]

    if not os.path.isfile(json_filename):
        print(f"File {json_filename} does not exist.")
        sys.exit(1)

    with open(json_filename, 'r') as json_file:
        json_data = json.load(json_file)


    #html_content = json_to_html(json_data)
    json_table = json_to_list_table(json_data)

    html_filename = os.path.splitext(json_filename)[0] + '.html'
    csv_filename = os.path.splitext(json_filename)[0] + '.csv'
    save_list_to_csv(json_table, csv_filename)

    with open(html_filename, 'w') as html_file:
        html_file.write(list_to_html_table(json_table))

    print(f"HTML file has been created: {html_filename}")

if __name__ == "__main__":
    main()
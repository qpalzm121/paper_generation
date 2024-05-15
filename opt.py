from vector_db import VectorDB


def generate_similar_paper(subject, grade, db_directory="./presist_timu_1", paper_path="./shijuan/shijuan_1.txt"):
    vector_db = VectorDB(subject, grade)
    questions = vector_db.split_paper()
    new_question_ids = vector_db.query_and_replace(questions)
    new_questions = vector_db.vector_to_text(new_question_ids)
    with open(f'./shijuan/new_paper_{subject}_{grade}.txt', 'w', encoding='utf-8') as f:
        for i, (_, question_text) in enumerate(new_questions, start=1):
            f.write(f'{i}. {question_text}\n')


subject = 1
grade = 9
generate_similar_paper(subject, grade)
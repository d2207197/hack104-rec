from elasticsearch_dsl import (
    connections,
    Document,
    Text, Keyword,
    Integer, Long
)


class Job(Document):
    custno = Keyword()
    jobno = Integer()
    description = Keyword() #
    job = Keyword() #
    # job = Text(index='not_analyzed')  # issue: https://github.com/elastic/elasticsearch/issues/21134
    others = Keyword() # 
    addr_no = Long()
    edu = Long()
    exp_jobcat1 = Long()
    exp_jobcat2 = Long()
    exp_jobcat3 = Long()
    industry = Long()
    jobcat1 = Long()
    jobcat2 = Long()
    jobcat3 = Long()
    language1 = Long()
    language2 = Long()
    language3 = Long()
    major_cat = Integer()
    major_cat2 = Integer()
    major_cat3 = Integer()
    need_emp = Long()
    need_emp1 = Long()
    period = Long()
    role = Long()
    role_status = Long()
    s2 = Long()
    s3 = Long()
    s9 = Long()
    salary_high = Long()
    salary_low = Long()
    startby = Long()
    worktime = Integer()
    invoice = Long()
    management = Keyword()  # 
    name = Keyword()  #
    product = Keyword()  #
    profile = Keyword()  #
    welfare = Keyword()  #

    class Index:
        name = 'hack104'


    @classmethod
    def from_row(cls, row):
        all_fields = row.__fields__
        token_fields = ['description', 'job', 'others', 'management', 'name', 'product', 'profile', 'welfare']
        pk_field = 'jobno'

        doc = cls()

        for field in all_fields:
            if field in token_fields:
                setattr(doc, field, getattr(getattr(row, field), 'token'))
            else:
                setattr(doc, field, getattr(row, field))

        doc.meta.id = pk_field
        
        return doc

    @classmethod
    def from_dict(cls, obj):
        token_fields = ['description', 'job', 'others', 'management', 'name', 'product', 'profile', 'welfare']
        pk_field = 'jobno'

        doc = cls()
        try:
            for field, body in obj.items():
                if field in token_fields:
                    setattr(doc, field, body.get('token'))
                else:
                    setattr(doc, field, body)
        except Exception as err:
            print('@@@@', 'field', field)
            print('@@@@', 'body', body)
            print('@@@@', obj)

            raise err

        doc.meta.id = obj.get(pk_field)
        
        return doc



if __name__ == '__main__':
    connections.create_connection(hosts=['localhost:9201'], timeout=20)
    print('Initiating index: `hack104` for type: `Job`')
    Job.init()

    import fileinput, json
    print('Press Enter to leave')
    for line in fileinput.input():
        if len(line.strip()) == 0:
            break
        print(json.dumps(Job.from_dict(json.loads(line)).to_dict(include_meta=True)))


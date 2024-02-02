from prefect import task, flow

@task
def switch_on():
    print("The switch has been turned on")


@flow(log_prints=True)
def runner():
    switch_on()


if __name__=="__main__":
    runner()

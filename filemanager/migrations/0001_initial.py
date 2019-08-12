# Generated by Django 2.2.4 on 2019-08-12 21:01

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Data',
            fields=[
                ('file_id', models.AutoField(primary_key=True, serialize=False)),
                ('file', models.FileField(max_length=255, null=True, upload_to='')),
                ('date_created', models.DateTimeField(default=django.utils.timezone.now)),
            ],
        ),
    ]

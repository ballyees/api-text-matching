import React, { useState } from "react";
import { Container, Center, VStack, CircularProgress, CircularProgressLabel, useBoolean, Text, Box } from '@chakra-ui/react'

const regx = new RegExp(/.*base64,/);
const url = 'http://127.0.0.1:8000/'
const fileFormat = ['doc', 'docx', 'txt']
export default function App() {
  const [textResponse, setTextResponse] = useState('');
  const [loading, setLoading] = useBoolean()

  const showFile = (e) => {
    if (e.target.files.length === 0) return;
    setLoading.on();
    e.preventDefault();
    window.fileName = e.target.files[0].name
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      let format = window.fileName.split('.')[1]
      postFile(text.replace(regx, ''), format).then(json_data => {
        let key = Object.keys(json_data)
        let text = ''
        for (let i in key) {
          text += json_data[key[i]] + '\n|\n';
        }
        setTextResponse(text);
        
        setLoading.off();
      })
    };
    
    reader.readAsDataURL(e.target.files[0]);
  };
  return (
    <Container color='gray.500' pt={10} maxW='container.lg'>
      <VStack>
        <Center>
          <Box>
            <input type="file" onChange={showFile} />
          </Box>
        </Center>
        <Center>
          <Box borderRadius='md' color='black.700'>
          {(loading) ?
            <CircularProgress isIndeterminate color='gray.900' size='200px' pt={5}>
              <CircularProgressLabel color='gray.600'>Loading</CircularProgressLabel>
            </CircularProgress>
            :
            <Text>{textResponse}</Text>
          }
          </Box>
        </Center>
      </VStack>
    </Container>
  )
}

async function postFile(data, format) {
  if(!search(format, fileFormat)){
    return {'response': 'file format not found!!!'}
  }
  const response = await fetch(url+format, {
    method: 'POST',
    body: JSON.stringify({ file: data })
  })
  return response.json(response);
}

function search(key, list){
  for(let i in list){
    if(list[i] === key){
      return true;
    }
  }
  return false
}